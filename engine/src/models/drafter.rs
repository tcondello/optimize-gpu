//! Co-Resident 100%-on-RTX-3080 Neural Speculative Drafter (`Qwen3-0.6B-Q8_0`)
//! & Patched-Trace Calibrated Reasoning N-Gram Prior (`Phase 5` Speculative Drafter Pairing).
//!
//! Zero external crates, zero runtime Docker, zero `.cu` kernel changes.
//! Shares `NvGpu` and `CubinModule` with the parent `QwenEngine` and feeds
//! 3-token draft continuations `[d0, d1, d2]` into the 100%-lossless `BM=4` megakernel verifier!

#![allow(dead_code)]

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::atomic::{fence, Ordering};
use std::time::{Duration, Instant};

use super::qwen_abi::{QwenDecodeCtrl, QwenLayerDesc, QwenModelDesc};
use crate::gguf::{
    decode_1d_f32, extract_quant_weight_bytes, precompute_freqs_cis_f32, resolve_model_path,
    GgufFile, ModelConfig,
};
use crate::hw::{put_u32, put_u64, CubinModule, HostMappedBuffer, NvGpu, VramBuffer};
use crate::lora::{prepare_lora_from_path, QwenLayerLoraDesc, LORA_ARENA_DEFAULT_BYTES};

const DRAFT_PREFILL_CHUNK: usize = 64;
const NGRAM_BIN_MAGIC: &[u8; 4] = b"QDNG";

// -----------------------------------------------------------------------------
// Patched-Trace Calibrated High-Precision N-Gram Prior (`5`-gram + `4`-gram)
// -----------------------------------------------------------------------------

fn json_esc(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

#[derive(Clone, Default)]
pub struct CalibratedNgramPrior {
    map5: HashMap<(i32, i32, i32, i32, i32), [i32; 3]>,
    map4: HashMap<(i32, i32, i32, i32), [i32; 3]>,
    source_path: String,
}

impl CalibratedNgramPrior {
    #[inline]
    pub fn counts(&self) -> (usize, usize) {
        (self.map5.len(), self.map4.len())
    }

    #[inline]
    pub fn source_path(&self) -> &str {
        &self.source_path
    }

    pub fn load_bin(path: &Path) -> Result<Self, String> {
        let b = fs::read(path)
            .map_err(|e| format!("Failed reading calibrated drafter prior {}: {}", path.display(), e))?;
        if b.len() < 16 || &b[0..4] != NGRAM_BIN_MAGIC {
            return Err(format!("Invalid QDNG header in {}", path.display()));
        }
        let ver = u32::from_le_bytes(b[4..8].try_into().unwrap());
        if ver != 1 {
            return Err(format!("Unsupported QDNG version {}", ver));
        }
        let n5 = u32::from_le_bytes(b[8..12].try_into().unwrap()) as usize;
        let n4 = u32::from_le_bytes(b[12..16].try_into().unwrap()) as usize;
        let need = 16 + n5 * 32 + n4 * 28;
        if b.len() < need {
            return Err(format!("Truncated QDNG file {} ({} < {})", path.display(), b.len(), need));
        }
        let mut pos = 16usize;
        let mut map5 = HashMap::with_capacity(n5 * 2 + 16);
        for _ in 0..n5 {
            let mut v = [0i32; 8];
            for item in &mut v {
                *item = i32::from_le_bytes(b[pos..pos + 4].try_into().unwrap());
                pos += 4;
            }
            map5.insert((v[0], v[1], v[2], v[3], v[4]), [v[5], v[6], v[7]]);
        }
        let mut map4 = HashMap::with_capacity(n4 * 2 + 16);
        for _ in 0..n4 {
            let mut v = [0i32; 7];
            for item in &mut v {
                *item = i32::from_le_bytes(b[pos..pos + 4].try_into().unwrap());
                pos += 4;
            }
            map4.insert((v[0], v[1], v[2], v[3]), [v[4], v[5], v[6]]);
        }
        Ok(Self {
            map5,
            map4,
            source_path: path.display().to_string(),
        })
    }

    pub fn save_bin(&self, path: &Path) -> Result<(), String> {
        if let Some(p) = path.parent() {
            if !p.as_os_str().is_empty() {
                fs::create_dir_all(p).map_err(|e| e.to_string())?;
            }
        }
        let mut out = Vec::with_capacity(16 + self.map5.len() * 32 + self.map4.len() * 28);
        out.extend_from_slice(NGRAM_BIN_MAGIC);
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&(self.map5.len() as u32).to_le_bytes());
        out.extend_from_slice(&(self.map4.len() as u32).to_le_bytes());
        for (&(k0, k1, k2, k3, k4), &[d0, d1, d2]) in &self.map5 {
            for x in [k0, k1, k2, k3, k4, d0, d1, d2] {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        for (&(k0, k1, k2, k3), &[d0, d1, d2]) in &self.map4 {
            for x in [k0, k1, k2, k3, d0, d1, d2] {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        fs::write(path, out).map_err(|e| e.to_string())
    }

    /// Distill high-confidence (`min_prob` & `min_support`) 3-token continuations directly from
    /// fresh token sequences generated by `./checkpoints/qwen-patched/`.
    pub fn distill_from_sequences(
        sequences: &[Vec<i32>],
        min_support: u32,
        min_prob: f32,
    ) -> Self {
        let mut counts5: HashMap<(i32, i32, i32, i32, i32), HashMap<[i32; 3], u32>> = HashMap::new();
        let mut counts4: HashMap<(i32, i32, i32, i32), HashMap<[i32; 3], u32>> = HashMap::new();

        for seq in sequences {
            if seq.len() < 8 {
                continue;
            }
            for i in 0..=(seq.len() - 8) {
                let k5 = (seq[i], seq[i + 1], seq[i + 2], seq[i + 3], seq[i + 4]);
                let d5 = [seq[i + 5], seq[i + 6], seq[i + 7]];
                *counts5.entry(k5).or_default().entry(d5).or_insert(0) += 1;
            }
            for i in 0..=(seq.len() - 7) {
                let k4 = (seq[i], seq[i + 1], seq[i + 2], seq[i + 3]);
                let d4 = [seq[i + 4], seq[i + 5], seq[i + 6]];
                *counts4.entry(k4).or_default().entry(d4).or_insert(0) += 1;
            }
        }

        let mut map5 = HashMap::new();
        for (k, dist) in counts5 {
            let total: u32 = dist.values().sum();
            if total < min_support {
                continue;
            }
            if let Some((&best_d, &best_c)) = dist.iter().max_by_key(|(_, &c)| c) {
                if (best_c as f32) / (total as f32) >= min_prob {
                    map5.insert(k, best_d);
                }
            }
        }

        let mut map4 = HashMap::new();
        for (k, dist) in counts4 {
            let total: u32 = dist.values().sum();
            if total < min_support + 1 {
                continue;
            }
            if let Some((&best_d, &best_c)) = dist.iter().max_by_key(|(_, &c)| c) {
                if (best_c as f32) / (total as f32) >= (min_prob + 0.05).min(0.98) {
                    map4.insert(k, best_d);
                }
            }
        }

        Self {
            map5,
            map4,
            source_path: String::new(),
        }
    }

    #[inline]
    pub fn lookup(&self, seq: &[i32]) -> Option<[i32; 3]> {
        let n = seq.len();
        if n >= 5 {
            let k5 = (seq[n - 5], seq[n - 4], seq[n - 3], seq[n - 2], seq[n - 1]);
            if let Some(&d) = self.map5.get(&k5) {
                return Some(d);
            }
        }
        if n >= 4 {
            let k4 = (seq[n - 4], seq[n - 3], seq[n - 2], seq[n - 1]);
            if let Some(&d) = self.map4.get(&k4) {
                return Some(d);
            }
        }
        None
    }
}

// -----------------------------------------------------------------------------
// Co-Resident 100%-on-RTX-3080 `Qwen3-0.6B-Q8_0` Neural Speculative Drafter
// -----------------------------------------------------------------------------

pub struct QwenNeuralDrafter {
    model_ref: String,
    cfg: ModelConfig,
    vram_bytes: u64,

    _owned_vram: Vec<VramBuffer>,
    layers_host: Vec<QwenLayerDesc>,
    layers_gpu: VramBuffer,
    model_desc_gpu: VramBuffer,

    lora_arena: Option<VramBuffer>,
    lora_descs_host: Vec<QwenLayerLoraDesc>,

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
    synced_tokens: Vec<i32>,
    calibrated_prior: Option<CalibratedNgramPrior>,
}

impl QwenNeuralDrafter {
    #[inline]
    pub fn model_ref(&self) -> &str {
        &self.model_ref
    }

    #[inline]
    pub fn has_prior(&self) -> bool {
        self.calibrated_prior.is_some()
    }

    pub fn set_prior(&mut self, prior: CalibratedNgramPrior) -> (usize, usize) {
        let counts = prior.counts();
        self.calibrated_prior = Some(prior);
        counts
    }

    pub fn load_prior(&mut self, prior_path: &Path) -> Result<(usize, usize), String> {
        let prior = CalibratedNgramPrior::load_bin(prior_path)?;
        let counts = prior.counts();
        self.calibrated_prior = Some(prior);
        Ok(counts)
    }

    pub fn status_json(&self, load_ms_opt: Option<f64>) -> String {
        let (n5, n4, ppath) = if let Some(ref pr) = self.calibrated_prior {
            let (c5, c4) = pr.counts();
            (c5, c4, pr.source_path())
        } else {
            (0, 0, "")
        };
        if let Some(load_ms) = load_ms_opt {
            format!(
                "{{\"active\":true,\"model_ref\":\"{}\",\"model_name\":\"{}\",\"vram_mb\":{:.2},\"prior_5grams\":{},\"prior_4grams\":{},\"load_ms\":{:.2}}}",
                json_esc(&self.model_ref),
                json_esc(&self.cfg.model_name),
                (self.vram_bytes as f64) / (1024.0 * 1024.0),
                n5,
                n4,
                load_ms
            )
        } else {
            format!(
                "{{\"active\":true,\"model_ref\":\"{}\",\"model_name\":\"{}\",\"vram_mb\":{:.2},\"prior_5grams\":{},\"prior_4grams\":{},\"prior_path\":\"{}\"}}",
                json_esc(&self.model_ref),
                json_esc(&self.cfg.model_name),
                (self.vram_bytes as f64) / (1024.0 * 1024.0),
                n5,
                n4,
                json_esc(ppath)
            )
        }
    }

    pub fn load(gpu: &mut NvGpu, draft_ref: &str, max_context: usize) -> Result<Self, String> {
        let t0 = Instant::now();
        let eff_ref = if draft_ref.trim().is_empty() {
            "qwen3:0.6b"
        } else {
            draft_ref.trim()
        };
        let local_path = resolve_model_path(eff_ref)?;
        let stem = local_path.file_stem().and_then(|s| s.to_str()).unwrap_or("qwen-draft");
        let gguf = GgufFile::open(&local_path)?;
        let tok = gguf.build_tokenizer()?;
        let cfg = ModelConfig::from_gguf(&gguf, &tok, max_context, stem)?;

        let mut owned_vram = Vec::<VramBuffer>::new();
        let mut total_vram = 0u64;
        let mut alloc_track = |g: &mut NvGpu, sz: u64| -> Result<VramBuffer, String> {
            let b = g.alloc_vram(sz.max(4096))?;
            total_vram += b.size;
            Ok(b)
        };

        let kv_alloc_ctx = cfg.max_ctx + DRAFT_PREFILL_CHUNK;

        let mut small_arena_bytes = Vec::<u8>::with_capacity(2 << 20);
        let push_f32_arena = |arena: &mut Vec<u8>, vals: &[f32]| -> u64 {
            let pad = (256 - (arena.len() % 256)) % 256;
            arena.resize(arena.len() + pad, 0);
            let off = arena.len() as u64;
            for &v in vals {
                arena.extend_from_slice(&v.to_le_bytes());
            }
            off
        };

        let freqs_vec = precompute_freqs_cis_f32(cfg.rope_dim, kv_alloc_ctx, cfg.rope_theta);
        let mut freqs_bytes = Vec::with_capacity(freqs_vec.len() * 4);
        for &v in &freqs_vec {
            freqs_bytes.extend_from_slice(&v.to_le_bytes());
        }
        let freqs_cis_gpu = alloc_track(gpu, freqs_bytes.len() as u64)?;
        gpu.upload_to_vram(&freqs_cis_gpu, 0, &freqs_bytes)?;

        let (emb_cow, emb_type, _, _) = extract_quant_weight_bytes(&gguf, "token_embd.weight")?;
        let emb_buf = alloc_track(gpu, emb_cow.len() as u64)?;
        gpu.upload_to_vram(&emb_buf, 0, &emb_cow)?;
        let emb_gpu_va = emb_buf.va_addr;
        owned_vram.push(emb_buf);

        let (lm_head_va, lm_head_type) = if gguf.has_tensor("output.weight") {
            let (lh_cow, lh_type, _, _) = extract_quant_weight_bytes(&gguf, "output.weight")?;
            let lh_buf = alloc_track(gpu, lh_cow.len() as u64)?;
            gpu.upload_to_vram(&lh_buf, 0, &lh_cow)?;
            let va = lh_buf.va_addr;
            owned_vram.push(lh_buf);
            (va, lh_type)
        } else {
            (emb_gpu_va, emb_type)
        };

        let out_norm_vec = decode_1d_f32(&gguf, "output_norm.weight")?
            .ok_or("Missing output_norm.weight in draft GGUF")?;
        let out_norm_rel = push_f32_arena(&mut small_arena_bytes, &out_norm_vec);

        let kv_layer_bytes = ((2 * cfg.n_kv_heads * kv_alloc_ctx * cfg.head_dim * 2) as u64 + 255) & !255;
        let kv_slab = alloc_track(gpu, kv_layer_bytes * (cfg.num_layers as u64))?;
        let kv_slab_va = kv_slab.va_addr;
        owned_vram.push(kv_slab);

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
                let buf = alloc_track(gpu, fused.len() as u64)?;
                gpu.upload_to_vram(&buf, 0, &fused)?;
                let va = buf.va_addr;
                owned_vram.push(buf);
                let brel = fuse_biases(&[(&bq, oq), (&bk, ok), (&bv, ov)])
                    .map(|v| push_f32_arena(&mut small_arena_bytes, &v));
                (va, tq, brel, 0u64, 0i32, None, 0i32)
            } else if tq == tk {
                let mut fused_qk = Vec::with_capacity(wq.len() + wk.len());
                fused_qk.extend_from_slice(&wq);
                fused_qk.extend_from_slice(&wk);
                let buf_qk = alloc_track(gpu, fused_qk.len() as u64)?;
                gpu.upload_to_vram(&buf_qk, 0, &fused_qk)?;
                let buf_v = alloc_track(gpu, wv.len() as u64)?;
                gpu.upload_to_vram(&buf_v, 0, &wv)?;
                let va_qk = buf_qk.va_addr;
                let va_v = buf_v.va_addr;
                owned_vram.push(buf_qk);
                owned_vram.push(buf_v);
                let bqk_rel = fuse_biases(&[(&bq, oq), (&bk, ok)])
                    .map(|v| push_f32_arena(&mut small_arena_bytes, &v));
                let bv_rel = bv.map(|v| push_f32_arena(&mut small_arena_bytes, &v));
                (va_qk, tq, bqk_rel, va_v, tv, bv_rel, 1i32)
            } else {
                return Err(format!("Draft layer {} mismatched Q/K types", i));
            };

            let (wo, to_t, _, _) = extract_quant_weight_bytes(&gguf, &format!("{}.attn_output.weight", p))?;
            let bo = decode_1d_f32(&gguf, &format!("{}.attn_output.bias", p))?;
            let buf_o = alloc_track(gpu, wo.len() as u64)?;
            gpu.upload_to_vram(&buf_o, 0, &wo)?;
            let w_o_va = buf_o.va_addr;
            owned_vram.push(buf_o);
            let b_o_rel = bo.map(|v| push_f32_arena(&mut small_arena_bytes, &v));

            let (wg, tg, _, _) = extract_quant_weight_bytes(&gguf, &format!("{}.ffn_gate.weight", p))?;
            let (wu, tu, _, _) = extract_quant_weight_bytes(&gguf, &format!("{}.ffn_up.weight", p))?;
            let (w_gu_va, gu_type, w_up_va, up_type, split_gu) = if tg == tu {
                let mut fused_gu = Vec::with_capacity(wg.len() + wu.len());
                fused_gu.extend_from_slice(&wg);
                fused_gu.extend_from_slice(&wu);
                let buf_gu = alloc_track(gpu, fused_gu.len() as u64)?;
                gpu.upload_to_vram(&buf_gu, 0, &fused_gu)?;
                let va = buf_gu.va_addr;
                owned_vram.push(buf_gu);
                (va, tg, 0u64, 0i32, 0i32)
            } else {
                let buf_g = alloc_track(gpu, wg.len() as u64)?;
                gpu.upload_to_vram(&buf_g, 0, &wg)?;
                let buf_u = alloc_track(gpu, wu.len() as u64)?;
                gpu.upload_to_vram(&buf_u, 0, &wu)?;
                let va_g = buf_g.va_addr;
                let va_u = buf_u.va_addr;
                owned_vram.push(buf_g);
                owned_vram.push(buf_u);
                (va_g, tg, va_u, tu, 1i32)
            };

            let (wd, td, _, _) = extract_quant_weight_bytes(&gguf, &format!("{}.ffn_down.weight", p))?;
            let buf_d = alloc_track(gpu, wd.len() as u64)?;
            gpu.upload_to_vram(&buf_d, 0, &wd)?;
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

        let small_arena_gpu = alloc_track(gpu, small_arena_bytes.len().max(4096) as u64)?;
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
        let layers_gpu = alloc_track(gpu, layers_bytes.len() as u64)?;
        gpu.upload_to_vram(&layers_gpu, 0, layers_bytes)?;
        let layers_ptr_va = layers_gpu.va_addr;

        let q_dim = cfg.n_heads * cfg.head_dim;
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        let max_k = cfg.dim.max(q_dim).max(cfg.hidden_dim);

        let ws_x_res = alloc_track(gpu, (DRAFT_PREFILL_CHUNK * cfg.dim * 4) as u64)?;
        let ws_x_res2 = alloc_track(gpu, (DRAFT_PREFILL_CHUNK * cfg.dim * 4) as u64)?;
        let ws_xq = alloc_track(gpu, (DRAFT_PREFILL_CHUNK * max_k) as u64)?;
        let ws_xds = alloc_track(gpu, (DRAFT_PREFILL_CHUNK * (max_k / 32) * 8) as u64)?;
        let ws_qkv = alloc_track(gpu, (DRAFT_PREFILL_CHUNK * (q_dim + 2 * kv_dim) * 4) as u64)?;
        let ws_v_sep = alloc_track(gpu, (DRAFT_PREFILL_CHUNK * kv_dim * 4) as u64)?;
        let ws_q_roped = alloc_track(gpu, (DRAFT_PREFILL_CHUNK * q_dim * 4) as u64)?;
        let ws_attn_out = alloc_track(gpu, (DRAFT_PREFILL_CHUNK * q_dim * 4) as u64)?;
        let ws_gate_up = alloc_track(gpu, (DRAFT_PREFILL_CHUNK * 2 * cfg.hidden_dim * 4) as u64)?;
        let ws_cta_val = alloc_track(gpu, 4 * 128 * 4)?;
        let ws_cta_idx = alloc_track(gpu, 4 * 128 * 4)?;
        let ws_bar_state = alloc_track(gpu, 4096)?;
        gpu.zero_vram(&ws_bar_state)?;

        let cta_val_va = ws_cta_val.va_addr;
        let cta_idx_va = ws_cta_idx.va_addr;
        owned_vram.push(ws_cta_val);
        owned_vram.push(ws_cta_idx);

        // Set stop_ids on the drafter to `-1` so a 3-step draft rollout never halts early at token 0
        // (the target BM=4 verifier enforces exact stop_ids losslessly!)
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
            stop_id0: -1,
            stop_id1: -1,
            stop_id2: -1,
            stop_id3: -1,
            ws_ext_embed: 0,
            ws_hidden_last: 0,
            ws_logits_out: 0,
            _pad_model: 0,
        };
        let mdesc_bytes = unsafe {
            std::slice::from_raw_parts(
                &mdesc as *const QwenModelDesc as *const u8,
                std::mem::size_of::<QwenModelDesc>(),
            )
        };
        let model_desc_gpu = alloc_track(gpu, 4096)?;
        gpu.upload_to_vram(&model_desc_gpu, 0, mdesc_bytes)?;

        let ctrl_buf = gpu.alloc_host_mapped(std::mem::size_of::<QwenDecodeCtrl>())?;
        let prefill_toks_buf = gpu.alloc_host_mapped(DRAFT_PREFILL_CHUNK * 4)?;
        let lora_descs_host = vec![QwenLayerLoraDesc::default(); cfg.num_layers];

        eprintln!(
            "[rtx drafter] Loaded co-resident neural drafter '{}' ({:.1} MB VRAM, {}L, dim={}, 100% RTX 3080) in {:.2}s",
            cfg.model_name,
            (total_vram as f64) / (1024.0 * 1024.0),
            cfg.num_layers,
            cfg.dim,
            t0.elapsed().as_secs_f64()
        );

        Ok(Self {
            model_ref: eff_ref.to_string(),
            cfg,
            vram_bytes: total_vram,
            _owned_vram: owned_vram,
            layers_host,
            layers_gpu,
            model_desc_gpu,
            lora_arena: None,
            lora_descs_host,
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
            synced_tokens: Vec::with_capacity(4096),
            calibrated_prior: None,
        })
    }

    pub fn load_lora(&mut self, gpu: &mut NvGpu, path: &str, scale: f32) -> Result<(), String> {
        let q_dim = self.cfg.n_heads * self.cfg.head_dim;
        let kv_dim = self.cfg.n_kv_heads * self.cfg.head_dim;
        let mut prep = prepare_lora_from_path(
            path,
            scale,
            self.cfg.dim,
            q_dim,
            kv_dim,
            self.cfg.hidden_dim,
            self.cfg.num_layers,
        )?;
        let needed = (prep.arena_bytes.len() as u64).max(LORA_ARENA_DEFAULT_BYTES);
        if self.lora_arena.as_ref().map_or(true, |b| b.size < needed) {
            self.lora_arena = Some(gpu.alloc_vram(needed)?);
        }
        let arena_va = self.lora_arena.as_ref().unwrap().va_addr;
        let dsize = std::mem::size_of::<QwenLayerLoraDesc>();
        for l in 0..self.cfg.num_layers {
            if prep.layer_descs[l].active_mask != 0 {
                for proj in 0..7 {
                    if (prep.layer_descs[l].active_mask & (1u32 << proj)) != 0 {
                        prep.layer_descs[l].projs[proj].a_f16 += arena_va;
                        prep.layer_descs[l].projs[proj].b_f16 += arena_va;
                    }
                }
                let off = l * dsize;
                let bytes = unsafe {
                    std::slice::from_raw_parts(&prep.layer_descs[l] as *const QwenLayerLoraDesc as *const u8, dsize)
                };
                prep.arena_bytes[off..off + dsize].copy_from_slice(bytes);
                self.layers_host[l].lora_ptr = arena_va + (off as u64);
            } else {
                self.layers_host[l].lora_ptr = 0;
            }
        }
        self.lora_descs_host = prep.layer_descs;
        gpu.upload_to_vram(self.lora_arena.as_ref().unwrap(), 0, &prep.arena_bytes)?;
        let layers_bytes = unsafe {
            std::slice::from_raw_parts(
                self.layers_host.as_ptr() as *const u8,
                self.layers_host.len() * std::mem::size_of::<QwenLayerDesc>(),
            )
        };
        gpu.upload_to_vram(&self.layers_gpu, 0, layers_bytes)?;
        self.synced_tokens.clear();
        Ok(())
    }

    pub fn reset_cache(&mut self) {
        self.synced_tokens.clear();
    }

    fn run_prefill_chunk(
        &mut self,
        gpu: &mut NvGpu,
        module: &CubinModule,
        chunk: &[i32],
        start_pos: usize,
    ) -> Result<(), String> {
        let t_len = chunk.len();
        if t_len == 0 {
            return Ok(());
        }
        unsafe {
            let dst = self.prefill_toks_buf.as_ptr() as *mut i32;
            for (i, &tok) in chunk.iter().enumerate() {
                std::ptr::write_volatile(dst.add(i), tok);
            }
        }
        fence(Ordering::SeqCst);

        let dim = self.cfg.dim as i32;
        let hidden_dim = self.cfg.hidden_dim as i32;
        let n_heads = self.cfg.n_heads as i32;
        let n_kv_heads = self.cfg.n_kv_heads as i32;
        let head_dim = self.cfg.head_dim as i32;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let max_ctx_i32 = (self.cfg.max_ctx + DRAFT_PREFILL_CHUNK) as i32;
        let m_i32 = t_len as i32;
        let grid_m = ((t_len + 7) / 8) as u32;

        let k_emb = module.kernel("qwen_prefill_embed")?;
        let k_rms = module.kernel("qwen_prefill_rmsnorm_q8_1")?;
        let k_gemm = module.kernel("qwen_prefill_gemm")?;
        let k_qknorm = module.kernel("qwen_prefill_qknorm_rope_kv")?;
        let k_attn = module.kernel("qwen_prefill_flash_gqa")?;
        let k_quant = module.kernel("qwen_prefill_quant_q8_1")?;
        let k_swiglu = module.kernel("qwen_prefill_swiglu_q8_1")?;

        let mut chain: Vec<(u64, usize)> = Vec::with_capacity(1 + self.cfg.num_layers * 9);

        let mut a_emb = [0u8; 32];
        put_u64(&mut a_emb, 0, self.ws_x_res.va_addr);
        put_u64(&mut a_emb, 8, self.emb_gpu_va);
        put_u64(&mut a_emb, 16, self.prefill_toks_buf.va_addr);
        put_u32(&mut a_emb, 24, dim as u32);
        put_u32(&mut a_emb, 28, self.emb_type as u32);
        chain.push(gpu.prepare_qmd(k_emb, (t_len as u32, 1, 1), (256, 1, 1), &a_emb, None));

        let eps_bits = self.cfg.norm_eps.to_bits();
        let attn_scale_bits = (1.0f32 / (head_dim as f32).sqrt()).to_bits();

        for l in 0..self.cfg.num_layers {
            let ld = self.layers_host[l];
            let is_last_layer = l + 1 == self.cfg.num_layers;

            let mut a_rms1 = [0u8; 40];
            put_u64(&mut a_rms1, 0, self.ws_xq.va_addr);
            put_u64(&mut a_rms1, 8, self.ws_xds.va_addr);
            put_u64(&mut a_rms1, 16, self.ws_x_res.va_addr);
            put_u64(&mut a_rms1, 24, ld.attn_norm_w);
            put_u32(&mut a_rms1, 32, dim as u32);
            put_u32(&mut a_rms1, 36, eps_bits);
            chain.push(gpu.prepare_qmd(k_rms, (t_len as u32, 1, 1), (256, 1, 1), &a_rms1, None));

            if ld.split_v == 0 {
                let qkv_n = q_dim + 2 * kv_dim;
                let mut a_qkv = [0u8; 64];
                put_u64(&mut a_qkv, 0, self.ws_qkv.va_addr);
                put_u64(&mut a_qkv, 8, ld.w_qkv_or_qk);
                put_u64(&mut a_qkv, 16, self.ws_xq.va_addr);
                put_u64(&mut a_qkv, 24, self.ws_xds.va_addr);
                put_u64(&mut a_qkv, 32, ld.b_qkv_or_qk);
                put_u64(&mut a_qkv, 40, 0);
                put_u32(&mut a_qkv, 48, m_i32 as u32);
                put_u32(&mut a_qkv, 52, qkv_n as u32);
                put_u32(&mut a_qkv, 56, dim as u32);
                put_u32(&mut a_qkv, 60, ld.qkv_type as u32);
                chain.push(gpu.prepare_qmd(
                    k_gemm,
                    (((qkv_n as u32) + 3) / 4, grid_m, 1),
                    (128, 1, 1),
                    &a_qkv,
                    None,
                ));
            } else {
                let qk_n = q_dim + kv_dim;
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
                chain.push(gpu.prepare_qmd(
                    k_gemm,
                    (((qk_n as u32) + 3) / 4, grid_m, 1),
                    (128, 1, 1),
                    &a_qk,
                    None,
                ));

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
                chain.push(gpu.prepare_qmd(
                    k_gemm,
                    (((kv_dim as u32) + 3) / 4, grid_m, 1),
                    (128, 1, 1),
                    &a_v,
                    None,
                ));
            }

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
            chain.push(gpu.prepare_qmd(
                k_qknorm,
                ((n_heads + 2 * n_kv_heads) as u32, t_len as u32, 1),
                (256, 1, 1),
                &a_rope,
                None,
            ));

            if is_last_layer {
                break;
            }

            let mut a_attn = [0u8; 52];
            put_u64(&mut a_attn, 0, self.ws_attn_out.va_addr);
            put_u64(&mut a_attn, 8, self.ws_q_roped.va_addr);
            put_u64(&mut a_attn, 16, ld.kv_cache);
            put_u32(&mut a_attn, 24, m_i32 as u32);
            put_u32(&mut a_attn, 28, start_pos as u32);
            put_u32(&mut a_attn, 32, n_heads as u32);
            put_u32(&mut a_attn, 36, n_kv_heads as u32);
            put_u32(&mut a_attn, 40, head_dim as u32);
            put_u32(&mut a_attn, 44, max_ctx_i32 as u32);
            put_u32(&mut a_attn, 48, attn_scale_bits);
            chain.push(gpu.prepare_qmd(
                k_attn,
                (n_heads as u32, t_len as u32, 1),
                (128, 1, 1),
                &a_attn,
                None,
            ));

            let o_blks = (t_len * (q_dim as usize)) / 32;
            let mut a_qo = [0u8; 28];
            put_u64(&mut a_qo, 0, self.ws_xq.va_addr);
            put_u64(&mut a_qo, 8, self.ws_xds.va_addr);
            put_u64(&mut a_qo, 16, self.ws_attn_out.va_addr);
            put_u32(&mut a_qo, 24, o_blks as u32);
            chain.push(gpu.prepare_qmd(
                k_quant,
                (((o_blks as u32) + 3) / 4, 1, 1),
                (128, 1, 1),
                &a_qo,
                None,
            ));

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
            chain.push(gpu.prepare_qmd(
                k_gemm,
                (((dim as u32) + 3) / 4, grid_m, 1),
                (128, 1, 1),
                &a_wo,
                None,
            ));

            let mut a_rms2 = [0u8; 40];
            put_u64(&mut a_rms2, 0, self.ws_xq.va_addr);
            put_u64(&mut a_rms2, 8, self.ws_xds.va_addr);
            put_u64(&mut a_rms2, 16, self.ws_x_res2.va_addr);
            put_u64(&mut a_rms2, 24, ld.ffn_norm_w);
            put_u32(&mut a_rms2, 32, dim as u32);
            put_u32(&mut a_rms2, 36, eps_bits);
            chain.push(gpu.prepare_qmd(k_rms, (t_len as u32, 1, 1), (256, 1, 1), &a_rms2, None));

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
            chain.push(gpu.prepare_qmd(
                k_gemm,
                (((gu_n as u32) + 3) / 4, grid_m, 1),
                (128, 1, 1),
                &a_gu,
                None,
            ));

            let ffn_blks = (t_len * (hidden_dim as usize)) / 32;
            let mut a_sw = [0u8; 32];
            put_u64(&mut a_sw, 0, self.ws_xq.va_addr);
            put_u64(&mut a_sw, 8, self.ws_xds.va_addr);
            put_u64(&mut a_sw, 16, self.ws_gate_up.va_addr);
            put_u32(&mut a_sw, 24, m_i32 as u32);
            put_u32(&mut a_sw, 28, hidden_dim as u32);
            chain.push(gpu.prepare_qmd(
                k_swiglu,
                (((ffn_blks as u32) + 3) / 4, 1, 1),
                (128, 1, 1),
                &a_sw,
                None,
            ));

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
            chain.push(gpu.prepare_qmd(
                k_gemm,
                (((dim as u32) + 3) / 4, grid_m, 1),
                (128, 1, 1),
                &a_dn,
                None,
            ));
        }

        gpu.launch_qmd_chain_sync(&chain, true, Duration::from_secs(20))
    }

    /// Draft 3 candidate tokens `[d0, d1, d2]` 100% on the RTX 3080 from `full_seq`!
    /// Automatically reuses valid KV cache entries (`0` prefill when 0..2 tokens were accepted
    /// on the previous step, `1`-token catchup when all 3 + bonus token were accepted!).
    pub fn draft_k3(
        &mut self,
        gpu: &mut NvGpu,
        module: &CubinModule,
        full_seq: &[i32],
    ) -> Result<[i32; 3], String> {
        let n = full_seq.len();
        if n == 0 || n + 3 >= self.cfg.max_ctx + DRAFT_PREFILL_CHUNK {
            return Err("draft_k3: sequence empty or exceeds draft context".into());
        }

        // Fast check against optional trace-calibrated high-precision N-gram prior first (`0` GPU cost!)
        if let Some(ref prior) = self.calibrated_prior {
            if let Some(d) = prior.lookup(full_seq) {
                return Ok(d);
            }
        }

        let prefix_target = &full_seq[..n - 1];
        let common = prefix_target
            .iter()
            .zip(self.synced_tokens.iter())
            .take_while(|(a, b)| a == b)
            .count();
        self.synced_tokens.truncate(common);

        let ctrl_ptr = self.ctrl_buf.as_ptr() as *mut QwenDecodeCtrl;
        let mut mk_args = [0u8; 16];
        put_u64(&mut mk_args, 0, self.model_desc_gpu.va_addr);
        put_u64(&mut mk_args, 8, self.ctrl_buf.va_addr);
        let k_dec = module.kernel("qwen_persistent_decode_megakernel")?;

        let mut pos = common;
        while pos < n - 1 {
            let gap = (n - 1) - pos;
            if gap <= 2 {
                // Fast 1-launch megakernel KV catchup (~1.1 ms instead of 252-QMD prefill!)
                let catch_tok = full_seq[pos];
                unsafe {
                    (*ctrl_ptr).start_token = catch_tok;
                    (*ctrl_ptr).start_pos = pos as i32;
                    (*ctrl_ptr).max_steps = 1;
                    (*ctrl_ptr).temperature = 0.0;
                    (*ctrl_ptr).rng_seed = 1;
                    (*ctrl_ptr).cancel_flag = 0;
                    (*ctrl_ptr).error_code = 0;
                    (*ctrl_ptr).steps_completed = 0;
                    (*ctrl_ptr).stop_reason = 0;
                    (*ctrl_ptr).export_logits = 0;
                }
                fence(Ordering::SeqCst);
                gpu.launch_kernel_sync(k_dec, (gpu.active_sms, 1, 1), (512, 1, 1), &mk_args, true, Duration::from_secs(10))?;
                self.synced_tokens.push(catch_tok);
                pos += 1;
            } else {
                let take = gap.min(DRAFT_PREFILL_CHUNK);
                self.run_prefill_chunk(gpu, module, &full_seq[pos..pos + take], pos)?;
                self.synced_tokens.extend_from_slice(&full_seq[pos..pos + take]);
                pos += take;
            }
        }

        let cur_tok = full_seq[n - 1];
        let cur_pos = (n - 1) as i32;
        unsafe {
            (*ctrl_ptr).start_token = cur_tok;
            (*ctrl_ptr).start_pos = cur_pos;
            (*ctrl_ptr).max_steps = 3;
            (*ctrl_ptr).temperature = 0.0;
            (*ctrl_ptr).rng_seed = 1;
            (*ctrl_ptr).cancel_flag = 0;
            (*ctrl_ptr).error_code = 0;
            (*ctrl_ptr).steps_completed = 0;
            (*ctrl_ptr).stop_reason = 0;
            (*ctrl_ptr).export_logits = 0;
        }
        fence(Ordering::SeqCst);

        gpu.launch_kernel_sync(k_dec, (gpu.active_sms, 1, 1), (512, 1, 1), &mk_args, true, Duration::from_secs(10))?;

        let (err, d0, d1, d2) = unsafe {
            (
                (*ctrl_ptr).error_code,
                (*ctrl_ptr).tokens_out[0],
                (*ctrl_ptr).tokens_out[1],
                (*ctrl_ptr).tokens_out[2],
            )
        };
        if err != 0 {
            let _ = gpu.zero_vram(&self.ws_bar_state);
            self.synced_tokens.clear();
            return Err(format!("Draft GPU watchdog tripped at {}", err));
        }

        // During `max_steps = 3`, `qwen_decode_megakernel` populated KV cache at:
        // `n - 1` (`cur_tok`), `n` (`d0`), and `n + 1` (`d1`)!
        self.synced_tokens.push(cur_tok);
        self.synced_tokens.push(d0);
        self.synced_tokens.push(d1);

        Ok([d0, d1, d2])
    }
}
