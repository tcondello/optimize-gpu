//! Zero-Crate Fused Dynamic Rank-`r` (`r <= 32`) LoRA Hot-Swap Loader & VRAM Arena Manager.
//! Supports HuggingFace PEFT (`adapter_model.safetensors` + `adapter_config.json`) with `< 1 ms` hot-swap
//! and `0.0%` overhead when inactive.

#![allow(dead_code)]

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::gguf::{f16_to_f32, f32_to_f16_bits, GgufFile};
use crate::hw::{get_u16, get_u32};
use crate::safetensors::SafeTensorsFile;

#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_NO_TRANS: i32 = 111;
const CBLAS_TRANS: i32 = 112;

pub const MAX_LORA_RANK: usize = 32;
pub const LORA_ARENA_DEFAULT_BYTES: u64 = 64 << 20; // 64 MB reusable VRAM slab

// Projection indices inside `QwenLayerLoraDesc::projs[7]`
pub const LORA_PROJ_Q: usize = 0;
pub const LORA_PROJ_K: usize = 1;
pub const LORA_PROJ_V: usize = 2;
pub const LORA_PROJ_O: usize = 3;
pub const LORA_PROJ_GATE: usize = 4;
pub const LORA_PROJ_UP: usize = 5;
pub const LORA_PROJ_DOWN: usize = 6;

#[repr(C, align(16))]
#[derive(Clone, Copy, Default)]
pub struct QwenProjLora {
    pub a_f16: u64,   // const half* [rank, k_cols] (with alpha/rank scale baked in)
    pub b_f16: u64,   // const half* [n_rows, rank]
    pub rank: i32,    // 0 if inactive, 1..=32 if active
    pub out_off: i32, // row offset in target buffer for T=1 (e.g. 0, q_dim, q_dim+kv_dim)
    pub n_rows: i32,  // output rows of this sub-projection
    pub k_cols: i32,  // input cols of this sub-projection
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Default)]
pub struct QwenLayerLoraDesc {
    pub projs: [QwenProjLora; 7],
    pub active_mask: u32, // bitmask of active projs (bits 0..=6)
    pub _pad0: u32,
    pub _pad1: u64,
}

#[derive(Clone, Debug)]
pub struct LoraStatus {
    pub active: bool,
    pub path: String,
    pub rank: usize,
    pub alpha: f32,
    pub scale: f32,
    pub num_adapted_tensors: usize,
    pub vram_bytes: usize,
    pub swap_ms: f64,
}

impl Default for LoraStatus {
    fn default() -> Self {
        Self {
            active: false,
            path: String::new(),
            rank: 0,
            alpha: 0.0,
            scale: 1.0,
            num_adapted_tensors: 0,
            vram_bytes: 0,
            swap_ms: 0.0,
        }
    }
}

fn parse_adapter_config_alpha_rank(dir: &Path) -> (Option<f32>, Option<usize>) {
    let cfg_path = dir.join("adapter_config.json");
    let Ok(txt) = fs::read_to_string(&cfg_path) else {
        return (None, None);
    };
    let find_num = |needle: &str| -> Option<f32> {
        let idx = txt.find(needle)?;
        let rest = &txt[idx + needle.len()..];
        let colon = rest.find(':')?;
        let after = rest[colon + 1..].trim_start();
        let end = after
            .find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-')
            .unwrap_or(after.len());
        after[..end].parse::<f32>().ok()
    };
    let alpha = find_num("\"lora_alpha\"");
    let rank = find_num("\"r\"").map(|v| v as usize);
    (alpha, rank)
}

/// Classify a PEFT / GGUF LoRA tensor name into `(layer_idx, proj_idx, is_b)`.
fn classify_lora_tensor_name(name: &str) -> Option<(usize, usize, bool)> {
    let lower = name.to_ascii_lowercase();
    let is_a = lower.contains("lora_a");
    let is_b = lower.contains("lora_b");
    if !is_a && !is_b {
        return None;
    }

    // Extract layer index after `layers.` or `blk.`
    let layer_idx = if let Some(p) = lower.find("layers.") {
        let rest = &lower[p + 7..];
        let end = rest.find('.').unwrap_or(rest.len());
        rest[..end].parse::<usize>().ok()?
    } else if let Some(p) = lower.find("blk.") {
        let rest = &lower[p + 4..];
        let end = rest.find('.').unwrap_or(rest.len());
        rest[..end].parse::<usize>().ok()?
    } else {
        return None;
    };

    let proj_idx = if lower.contains("q_proj") || lower.contains("attn_q.") {
        LORA_PROJ_Q
    } else if lower.contains("k_proj") || lower.contains("attn_k.") {
        LORA_PROJ_K
    } else if lower.contains("v_proj") || lower.contains("attn_v.") {
        LORA_PROJ_V
    } else if lower.contains("o_proj") || lower.contains("attn_output.") {
        LORA_PROJ_O
    } else if lower.contains("gate_proj") || lower.contains("ffn_gate.") {
        LORA_PROJ_GATE
    } else if lower.contains("up_proj") || lower.contains("ffn_up.") {
        LORA_PROJ_UP
    } else if lower.contains("down_proj") || lower.contains("ffn_down.") {
        LORA_PROJ_DOWN
    } else {
        return None;
    };

    Some((layer_idx, proj_idx, is_b))
}

pub struct PreparedLoraPayload {
    pub arena_bytes: Vec<u8>,
    pub layer_descs: Vec<QwenLayerLoraDesc>,
    pub rank: usize,
    pub alpha: f32,
    pub scale: f32,
    pub num_adapted_tensors: usize,
}

/// Load a PEFT LoRA directory or `.safetensors` file and pack all `(A_scaled, B)` FP16 matrices
/// + `[num_layers]` `QwenLayerLoraDesc` descriptors into a single contiguous byte buffer ready for 1-DMA upload!
pub fn prepare_lora_from_path(
    path_str: &str,
    user_scale: f32,
    num_layers: usize,
    dim: usize,
    hidden_dim: usize,
    q_dim: usize,
    kv_dim: usize,
) -> Result<PreparedLoraPayload, String> {
    assert_eq!(std::mem::size_of::<QwenProjLora>(), 32);
    assert_eq!(std::mem::size_of::<QwenLayerLoraDesc>(), 240);

    let p = Path::new(path_str);
    let (st_path, cfg_dir): (PathBuf, Option<&Path>) = if p.is_dir() {
        let cand = p.join("adapter_model.safetensors");
        if !cand.exists() {
            return Err(format!("Missing adapter_model.safetensors inside {}", p.display()));
        }
        (cand, Some(p))
    } else if p.exists() {
        (p.to_path_buf(), p.parent())
    } else {
        return Err(format!("LoRA path not found: {}", path_str));
    };

    let (cfg_alpha, cfg_rank) = cfg_dir.map(parse_adapter_config_alpha_rank).unwrap_or((None, None));
    let st = SafeTensorsFile::open(&st_path)?;

    let mut pairs: HashMap<(usize, usize), (Option<String>, Option<String>)> = HashMap::new();
    for name in st.tensor_names() {
        if let Some((l, proj, is_b)) = classify_lora_tensor_name(&name) {
            if l < num_layers {
                let slot = pairs.entry((l, proj)).or_insert((None, None));
                if is_b {
                    slot.1 = Some(name);
                } else {
                    slot.0 = Some(name);
                }
            }
        }
    }

    if pairs.is_empty() {
        return Err(format!("No matching Qwen LoRA tensors found in {}", st_path.display()));
    }

    let expected_dims = |proj: usize| -> (usize, usize, i32) {
        match proj {
            LORA_PROJ_Q => (q_dim, dim, 0),
            LORA_PROJ_K => (kv_dim, dim, q_dim as i32),
            LORA_PROJ_V => (kv_dim, dim, (q_dim + kv_dim) as i32),
            LORA_PROJ_O => (dim, q_dim, 0),
            LORA_PROJ_GATE => (hidden_dim, dim, 0),
            LORA_PROJ_UP => (hidden_dim, dim, hidden_dim as i32),
            LORA_PROJ_DOWN => (dim, hidden_dim, 0),
            _ => unreachable!(),
        }
    };

    // Reserve head of arena for `[num_layers]` `QwenLayerLoraDesc` (aligned to 256B)
    let desc_bytes = num_layers * std::mem::size_of::<QwenLayerLoraDesc>();
    let mut arena = vec![0u8; (desc_bytes + 255) & !255];
    let mut layer_descs = vec![QwenLayerLoraDesc::default(); num_layers];

    let mut detected_rank = cfg_rank.unwrap_or(0);
    let mut adapted_count = 0usize;

    let mut keys: Vec<(usize, usize)> = pairs.keys().copied().collect();
    keys.sort_unstable();

    for (l, proj) in keys {
        let (Some(ref key_a), Some(ref key_b)) = pairs[&(l, proj)] else {
            continue;
        };
        let (exp_n, exp_k, out_off) = expected_dims(proj);
        let (a_decoded, a_shape) = st.decode_f32(key_a)?;
        let (b_decoded, b_shape) = st.decode_f32(key_b)?;
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(format!("LoRA layer {} proj {} is not 2D", l, proj));
        }

        let (a_f32_raw, r) = if a_shape[1] == exp_k {
            (a_decoded, a_shape[0])
        } else if a_shape[0] == exp_k {
            let r = a_shape[1];
            let mut tr = vec![0.0f32; r * exp_k];
            for kk in 0..exp_k {
                for rr in 0..r {
                    tr[rr * exp_k + kk] = a_decoded[kk * r + rr];
                }
            }
            (tr, r)
        } else {
            return Err(format!(
                "LoRA A shape {:?} mismatches expected K={} on layer {} proj {}",
                a_shape, exp_k, l, proj
            ));
        };

        if r == 0 || r > MAX_LORA_RANK {
            return Err(format!(
                "LoRA rank {} exceeds max fused SMEM rank {} (layer {} proj {})",
                r, MAX_LORA_RANK, l, proj
            ));
        }
        if detected_rank == 0 {
            detected_rank = r;
        }

        let b_f32 = if b_shape == [exp_n, r] {
            b_decoded
        } else if b_shape == [r, exp_n] {
            let mut tr = vec![0.0f32; exp_n * r];
            for rr in 0..r {
                for nn in 0..exp_n {
                    tr[nn * r + rr] = b_decoded[rr * exp_n + nn];
                }
            }
            tr
        } else {
            return Err(format!(
                "LoRA B shape {:?} mismatches expected [{}, {}] on layer {} proj {}",
                b_shape, exp_n, r, l, proj
            ));
        };

        let alpha = cfg_alpha.unwrap_or(r as f32);
        let eff_scale = (alpha / (r as f32)) * user_scale;

        let pad_a = (256 - (arena.len() % 256)) % 256;
        arena.resize(arena.len() + pad_a, 0);
        let off_a = arena.len() as u64;
        for &v in &a_f32_raw {
            arena.extend_from_slice(&f32_to_f16_bits(v * eff_scale).to_le_bytes());
        }

        let pad_b = (256 - (arena.len() % 256)) % 256;
        arena.resize(arena.len() + pad_b, 0);
        let off_b = arena.len() as u64;
        for &v in &b_f32 {
            arena.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
        }

        // Store relative offsets in `a_f16`/`b_f16` for patching with `lora_arena_va` on upload
        layer_descs[l].projs[proj] = QwenProjLora {
            a_f16: off_a,
            b_f16: off_b,
            rank: r as i32,
            out_off,
            n_rows: exp_n as i32,
            k_cols: exp_k as i32,
        };
        layer_descs[l].active_mask |= 1u32 << proj;
        adapted_count += 2;
    }

    let final_alpha = cfg_alpha.unwrap_or(detected_rank as f32);
    let final_scale = if detected_rank > 0 {
        (final_alpha / (detected_rank as f32)) * user_scale
    } else {
        user_scale
    };

    Ok(PreparedLoraPayload {
        arena_bytes: arena,
        layer_descs,
        rank: detected_rank,
        alpha: final_alpha,
        scale: final_scale,
        num_adapted_tensors: adapted_count,
    })
}

/// Generate a valid deterministic Rank-`r` HuggingFace PEFT `.safetensors` + `adapter_config.json`
/// test adapter for `q_proj` and `v_proj` across all layers (used by `rtx lora synth-test`).
pub fn write_synthetic_peft_adapter(
    out_dir: &Path,
    num_layers: usize,
    dim: usize,
    q_dim: usize,
    kv_dim: usize,
    rank: usize,
    alpha: f32,
    strength: f32,
) -> Result<(), String> {
    fs::create_dir_all(out_dir).map_err(|e| e.to_string())?;
    let cfg_json = format!(
        "{{\n  \"peft_type\": \"LORA\",\n  \"r\": {},\n  \"lora_alpha\": {:.1},\n  \"target_modules\": [\"q_proj\", \"v_proj\"]\n}}\n",
        rank, alpha
    );
    fs::write(out_dir.join("adapter_config.json"), cfg_json).map_err(|e| e.to_string())?;

    let mut body = Vec::<u8>::new();
    let mut header_entries = Vec::<String>::new();
    let mut rng: u64 = 0x517cc1b727220a95;
    let mut next_f16 = |scale: f32| -> u16 {
        rng ^= rng >> 12;
        rng ^= rng << 25;
        rng ^= rng >> 27;
        let u = (rng.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32 / 16777216.0;
        f32_to_f16_bits((u * 2.0 - 1.0) * scale)
    };

    for l in 0..num_layers {
        for (pname, n_out) in [("q_proj", q_dim), ("v_proj", kv_dim)] {
            // A: [rank, dim]
            let a_start = body.len();
            for _ in 0..(rank * dim) {
                body.extend_from_slice(&next_f16(0.02).to_le_bytes());
            }
            let a_end = body.len();
            header_entries.push(format!(
                "\"base_model.model.model.layers.{}.self_attn.{}.lora_A.weight\":{{\"dtype\":\"F16\",\"shape\":[{},{}],\"data_offsets\":[{},{}]}}",
                l, pname, rank, dim, a_start, a_end
            ));

            // B: [n_out, rank]
            let b_start = body.len();
            for _ in 0..(n_out * rank) {
                body.extend_from_slice(&next_f16(strength).to_le_bytes());
            }
            let b_end = body.len();
            header_entries.push(format!(
                "\"base_model.model.model.layers.{}.self_attn.{}.lora_B.weight\":{{\"dtype\":\"F16\",\"shape\":[{},{}],\"data_offsets\":[{},{}]}}",
                l, pname, n_out, rank, b_start, b_end
            ));
        }
    }

    let mut hdr_json = format!("{{{}}}", header_entries.join(","));
    while hdr_json.len() % 8 != 0 {
        hdr_json.push(' ');
    }
    let mut file_bytes = Vec::with_capacity(8 + hdr_json.len() + body.len());
    file_bytes.extend_from_slice(&(hdr_json.len() as u64).to_le_bytes());
    file_bytes.extend_from_slice(hdr_json.as_bytes());
    file_bytes.extend_from_slice(&body);

    fs::write(out_dir.join("adapter_model.safetensors"), file_bytes).map_err(|e| e.to_string())
}

// -----------------------------------------------------------------------------
// Offline Same-Ancestor Truncated SVD LoRA Extractor (`W_target - W_base -> A, B`)
// -----------------------------------------------------------------------------

/// Dequantize a 2D GGUF weight tensor (`Q8_0=8`, `Q4_K=12`, `Q6_K=14`, `F32=0`, `F16=1`, `BF16=30`)
/// into row-major `[out_rows, in_cols]` `f32` matching our CUDA kernel dequantization bit-for-bit.
pub fn dequant_gguf_2d_f32(gguf: &GgufFile, name: &str) -> Result<(Vec<f32>, usize, usize), String> {
    let info = gguf.req_tensor_info(name)?;
    if info.dims.len() != 2 {
        return Err(format!("Expected 2D tensor for '{}', got {:?}", name, info.dims));
    }
    let k_cols = info.dims[0];
    let n_rows = info.dims[1];
    let nel = n_rows * k_cols;
    let mut out = vec![0.0f32; nel];

    match info.ggml_type {
        0 => {
            let raw = gguf.tensor_bytes(info, nel * 4);
            for i in 0..nel {
                out[i] = f32::from_bits(get_u32(raw, i * 4));
            }
        }
        1 => {
            let raw = gguf.tensor_bytes(info, nel * 2);
            for i in 0..nel {
                out[i] = f16_to_f32(get_u16(raw, i * 2));
            }
        }
        30 => {
            let raw = gguf.tensor_bytes(info, nel * 2);
            for i in 0..nel {
                out[i] = f32::from_bits((get_u16(raw, i * 2) as u32) << 16);
            }
        }
        8 => {
            // Q8_0: 32 elements -> 34 bytes (f16 d + 32 i8)
            let n_blk = nel / 32;
            let raw = gguf.tensor_bytes(info, n_blk * 34);
            for b in 0..n_blk {
                let blk = &raw[b * 34..(b + 1) * 34];
                let d = f16_to_f32(get_u16(blk, 0));
                let dst = &mut out[b * 32..(b + 1) * 32];
                for j in 0..32 {
                    dst[j] = d * ((blk[2 + j] as i8) as f32);
                }
            }
        }
        12 => {
            // Q4_K: 256 elements -> 144 bytes (f16 d, f16 dmin, 12B scales/mins, 128B qs)
            let n_sb = nel / 256;
            let raw = gguf.tensor_bytes(info, n_sb * 144);
            for sb in 0..n_sb {
                let blk = &raw[sb * 144..(sb + 1) * 144];
                let d = f16_to_f32(get_u16(blk, 0));
                let dmin = f16_to_f32(get_u16(blk, 2));
                let sc = &blk[4..16];
                let qs = &blk[16..144];
                let dst = &mut out[sb * 256..(sb + 1) * 256];
                for chunk in 0..4usize {
                    let (sc_lo, mn_lo) = if chunk < 2 {
                        (sc[chunk * 2] & 63, sc[chunk * 2 + 4] & 63)
                    } else {
                        let j = chunk * 2;
                        (
                            (sc[j + 4] & 0x0f) | ((sc[j - 4] >> 6) << 4),
                            (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4),
                        )
                    };
                    let (sc_hi, mn_hi) = if chunk < 2 {
                        (sc[chunk * 2 + 1] & 63, sc[chunk * 2 + 5] & 63)
                    } else {
                        let j = chunk * 2 + 1;
                        (
                            (sc[j + 4] & 0x0f) | ((sc[j - 4] >> 6) << 4),
                            (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4),
                        )
                    };
                    let d1 = d * (sc_lo as f32);
                    let m1 = dmin * (mn_lo as f32);
                    let d2 = d * (sc_hi as f32);
                    let m2 = dmin * (mn_hi as f32);
                    let q_chunk = &qs[chunk * 32..(chunk + 1) * 32];
                    for l in 0..32usize {
                        dst[chunk * 64 + l] = d1 * ((q_chunk[l] & 0x0f) as f32) - m1;
                        dst[chunk * 64 + 32 + l] = d2 * ((q_chunk[l] >> 4) as f32) - m2;
                    }
                }
            }
        }
        14 => {
            // Q6_K: 256 elements -> 210 bytes (128B ql, 64B qh, 16B scales i8, 2B f16 d)
            let n_sb = nel / 256;
            let raw = gguf.tensor_bytes(info, n_sb * 210);
            for sb in 0..n_sb {
                let blk = &raw[sb * 210..(sb + 1) * 210];
                let ql = &blk[0..128];
                let qh = &blk[128..192];
                let sc = &blk[192..208];
                let d = f16_to_f32(get_u16(blk, 208));
                let dst = &mut out[sb * 256..(sb + 1) * 256];
                for n128 in 0..2usize {
                    let ql_p = &ql[n128 * 64..(n128 + 1) * 64];
                    let qh_p = &qh[n128 * 32..(n128 + 1) * 32];
                    let sc_p = &sc[n128 * 8..(n128 + 1) * 8];
                    let dst_p = &mut dst[n128 * 128..(n128 + 1) * 128];
                    for l in 0..32usize {
                        let is = l / 16;
                        let q1 = ((ql_p[l] & 0x0f) | (((qh_p[l] >> 0) & 3) << 4)) as i32 - 32;
                        let q2 = ((ql_p[l + 32] & 0x0f) | (((qh_p[l] >> 2) & 3) << 4)) as i32 - 32;
                        let q3 = ((ql_p[l] >> 4) | (((qh_p[l] >> 4) & 3) << 4)) as i32 - 32;
                        let q4 = ((ql_p[l + 32] >> 4) | (((qh_p[l] >> 6) & 3) << 4)) as i32 - 32;
                        dst_p[l] = d * ((sc_p[is] as i8) as f32) * (q1 as f32);
                        dst_p[l + 32] = d * ((sc_p[is + 2] as i8) as f32) * (q2 as f32);
                        dst_p[l + 64] = d * ((sc_p[is + 4] as i8) as f32) * (q3 as f32);
                        dst_p[l + 96] = d * ((sc_p[is + 6] as i8) as f32) * (q4 as f32);
                    }
                }
            }
        }
        other => return Err(format!("Unsupported ggml_type={} on '{}'", other, name)),
    }
    Ok((out, n_rows, k_cols))
}

/// Orthonormalize columns of row-major `[rows, cols]` matrix `m` via two-pass Modified Gram-Schmidt.
fn orthonormalize_cols(m: &mut [f32], rows: usize, cols: usize) {
    for j in 0..cols {
        for _pass in 0..2 {
            for i in 0..j {
                let mut dot = 0.0f64;
                for r in 0..rows {
                    dot += (m[r * cols + i] as f64) * (m[r * cols + j] as f64);
                }
                let d32 = dot as f32;
                for r in 0..rows {
                    m[r * cols + j] -= d32 * m[r * cols + i];
                }
            }
        }
        let mut norm_sq = 0.0f64;
        for r in 0..rows {
            let v = m[r * cols + j] as f64;
            norm_sq += v * v;
        }
        let inv = if norm_sq > 1e-20 { 1.0 / norm_sq.sqrt() } else { 0.0 } as f32;
        for r in 0..rows {
            m[r * cols + j] *= inv;
        }
    }
}

/// Cyclic Jacobi eigensolver for small symmetric `[p, p]` (`p <= 48`) matrix `a` (`a = V * diag(eig) * V^T`),
/// returning eigenvalues sorted descending and corresponding orthonormal eigenvectors `v_cols [p, p]` (column-major cols in row-major `[p, p]`).
fn jacobi_sym_eigen(mut a: Vec<f64>, p: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0.0f64; p * p];
    for i in 0..p {
        v[i * p + i] = 1.0;
    }
    for _sweep in 0..40 {
        let mut max_off = 0.0f64;
        for i in 0..p {
            for j in (i + 1)..p {
                max_off = max_off.max(a[i * p + j].abs());
            }
        }
        if max_off < 1e-13 {
            break;
        }
        for i in 0..p {
            for j in (i + 1)..p {
                let aij = a[i * p + j];
                if aij.abs() < 1e-15 {
                    continue;
                }
                let aii = a[i * p + i];
                let ajj = a[j * p + j];
                let tau = (ajj - aii) / (2.0 * aij);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                a[i * p + i] = aii - t * aij;
                a[j * p + j] = ajj + t * aij;
                a[i * p + j] = 0.0;
                a[j * p + i] = 0.0;

                for k in 0..p {
                    if k != i && k != j {
                        let aki = a[k * p + i];
                        let akj = a[k * p + j];
                        a[k * p + i] = c * aki - s * akj;
                        a[i * p + k] = a[k * p + i];
                        a[k * p + j] = s * aki + c * akj;
                        a[j * p + k] = a[k * p + j];
                    }
                    let vki = v[k * p + i];
                    let vkj = v[k * p + j];
                    v[k * p + i] = c * vki - s * vkj;
                    v[k * p + j] = s * vki + c * vkj;
                }
            }
        }
    }

    let mut order: Vec<usize> = (0..p).collect();
    order.sort_unstable_by(|&i, &j| {
        a[j * p + j]
            .partial_cmp(&a[i * p + i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut eigs = vec![0.0f64; p];
    let mut v_sorted = vec![0.0f64; p * p];
    for (new_c, &old_c) in order.iter().enumerate() {
        eigs[new_c] = a[old_c * p + old_c].max(0.0);
        for r in 0..p {
            v_sorted[r * p + new_c] = v[r * p + old_c];
        }
    }
    (eigs, v_sorted)
}

/// Compute Halko-Martinsson-Tropp Randomized Truncated Rank-`rank` SVD (`q = 2` power iterations)
/// of `delta [n, k]` using Apple `Accelerate` `cblas_sgemm`, returning `(lora_a [rank, k], lora_b [n, rank], captured_energy, total_energy)`.
fn randomized_svd_rank_r(
    delta: &[f32],
    n: usize,
    k: usize,
    rank: usize,
    seed: u64,
) -> (Vec<f32>, Vec<f32>, f64, f64) {
    let mut total_energy = 0.0f64;
    for &x in delta {
        let xd = x as f64;
        total_energy += xd * xd;
    }
    if total_energy < 1e-20 {
        return (vec![0.0f32; rank * k], vec![0.0f32; n * rank], 0.0, 0.0);
    }

    let p = (rank + 8).min(n).min(k);
    let mut rng = seed ^ 0x9e37_79b9_7f4a_7c15;
    let mut omega = vec![0.0f32; k * p];
    for v in &mut omega {
        rng ^= rng >> 12;
        rng ^= rng << 25;
        rng ^= rng >> 27;
        let u = ((rng.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32) / 16_777_216.0;
        *v = u * 2.0 - 1.0;
    }

    let mut q_mat = vec![0.0f32; n * p];
    let mut z_mat = vec![0.0f32; k * p];

    unsafe {
        // Y = D @ Omega -> [n, p]
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            n as i32,
            p as i32,
            k as i32,
            1.0,
            delta.as_ptr(),
            k as i32,
            omega.as_ptr(),
            p as i32,
            0.0,
            q_mat.as_mut_ptr(),
            p as i32,
        );
        orthonormalize_cols(&mut q_mat, n, p);

        // 2 subspace power iterations for tight singular spectrum accuracy
        for _ in 0..2 {
            // Z = D^T @ Q -> [k, p]
            cblas_sgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_TRANS,
                CBLAS_NO_TRANS,
                k as i32,
                p as i32,
                n as i32,
                1.0,
                delta.as_ptr(),
                k as i32,
                q_mat.as_ptr(),
                p as i32,
                0.0,
                z_mat.as_mut_ptr(),
                p as i32,
            );
            orthonormalize_cols(&mut z_mat, k, p);

            // Q = D @ Z -> [n, p]
            cblas_sgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_NO_TRANS,
                n as i32,
                p as i32,
                k as i32,
                1.0,
                delta.as_ptr(),
                k as i32,
                z_mat.as_ptr(),
                p as i32,
                0.0,
                q_mat.as_mut_ptr(),
                p as i32,
            );
            orthonormalize_cols(&mut q_mat, n, p);
        }
    }

    // B_small = Q^T @ D -> [p, k]
    let mut b_small = vec![0.0f32; p * k];
    let mut cov_f32 = vec![0.0f32; p * p];
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_TRANS,
            CBLAS_NO_TRANS,
            p as i32,
            k as i32,
            n as i32,
            1.0,
            q_mat.as_ptr(),
            p as i32,
            delta.as_ptr(),
            k as i32,
            0.0,
            b_small.as_mut_ptr(),
            k as i32,
        );
        // cov = B_small @ B_small^T -> [p, p]
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            p as i32,
            p as i32,
            k as i32,
            1.0,
            b_small.as_ptr(),
            k as i32,
            b_small.as_ptr(),
            k as i32,
            0.0,
            cov_f32.as_mut_ptr(),
            p as i32,
        );
    }

    let cov_f64: Vec<f64> = cov_f32.iter().map(|&v| v as f64).collect();
    let (eigs, u_small_64) = jacobi_sym_eigen(cov_f64, p);
    let r_eff = rank.min(p);

    let captured_energy: f64 = eigs[..r_eff].iter().sum();

    // U_r [p, r_eff] with balanced `sqrt(sigma)` / `1/sqrt(sigma)` scaling so FP16 never underflows!
    let mut u_b_scale = vec![0.0f32; p * r_eff];
    let mut u_a_scale = vec![0.0f32; r_eff * p];
    for j in 0..r_eff {
        let sigma = eigs[j].max(0.0).sqrt();
        let sb = if sigma > 1e-12 { sigma.sqrt() as f32 } else { 0.0 };
        let sa = if sigma > 1e-12 { (1.0 / sigma.sqrt()) as f32 } else { 0.0 };
        for row in 0..p {
            let uv = u_small_64[row * p + j] as f32;
            u_b_scale[row * r_eff + j] = uv * sb;
            u_a_scale[j * p + row] = uv * sa;
        }
    }

    let mut lora_b = vec![0.0f32; n * rank];
    let mut lora_a = vec![0.0f32; rank * k];
    unsafe {
        // lora_b[:, :r_eff] = Q [n, p] @ u_b_scale [p, r_eff]
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            n as i32,
            r_eff as i32,
            p as i32,
            1.0,
            q_mat.as_ptr(),
            p as i32,
            u_b_scale.as_ptr(),
            r_eff as i32,
            0.0,
            lora_b.as_mut_ptr(),
            rank as i32,
        );
        // lora_a[:r_eff, :] = u_a_scale [r_eff, p] @ B_small [p, k]
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            r_eff as i32,
            k as i32,
            p as i32,
            1.0,
            u_a_scale.as_ptr(),
            p as i32,
            b_small.as_ptr(),
            k as i32,
            0.0,
            lora_a.as_mut_ptr(),
            k as i32,
        );
    }

    (lora_a, lora_b, captured_energy, total_energy)
}

#[derive(Clone, Debug)]
pub struct SvdExtractStats {
    pub num_layers: usize,
    pub rank: usize,
    pub include_mlp: bool,
    pub num_tensors: usize,
    pub safetensors_bytes: usize,
    pub attn_energy_pct: f64,
    pub mlp_energy_pct: f64,
    pub total_energy_pct: f64,
    pub elapsed_s: f64,
}

/// Extract a HuggingFace PEFT Rank-`rank` LoRA adapter (`adapter_model.safetensors` + `adapter_config.json`)
/// directly from `(target_gguf - base_gguf)` for same-ancestor models (e.g. `Qwen3-Embedding-4B - Qwen3-4B`).
pub fn extract_svd_peft_adapter(
    base_gguf_path: &Path,
    target_gguf_path: &Path,
    out_dir: &Path,
    num_layers: usize,
    rank: usize,
    include_mlp: bool,
) -> Result<SvdExtractStats, String> {
    if rank == 0 || rank > MAX_LORA_RANK {
        return Err(format!("Rank {} must be in 1..={}", rank, MAX_LORA_RANK));
    }
    let t0 = std::time::Instant::now();
    let g_base = GgufFile::open(base_gguf_path)?;
    let g_targ = GgufFile::open(target_gguf_path)?;

    fs::create_dir_all(out_dir).map_err(|e| e.to_string())?;
    let targets_json = if include_mlp {
        "[\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"gate_proj\", \"up_proj\", \"down_proj\"]"
    } else {
        "[\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\"]"
    };
    let cfg_json = format!(
        "{{\n  \"peft_type\": \"LORA\",\n  \"r\": {},\n  \"lora_alpha\": {:.1},\n  \"target_modules\": {}\n}}\n",
        rank, rank as f32, targets_json
    );
    fs::write(out_dir.join("adapter_config.json"), cfg_json).map_err(|e| e.to_string())?;

    let specs: &[(&str, &str, bool)] = if include_mlp {
        &[
            ("attn_q.weight", "self_attn.q_proj", false),
            ("attn_k.weight", "self_attn.k_proj", false),
            ("attn_v.weight", "self_attn.v_proj", false),
            ("attn_output.weight", "self_attn.o_proj", false),
            ("ffn_gate.weight", "mlp.gate_proj", true),
            ("ffn_up.weight", "mlp.up_proj", true),
            ("ffn_down.weight", "mlp.down_proj", true),
        ]
    } else {
        &[
            ("attn_q.weight", "self_attn.q_proj", false),
            ("attn_k.weight", "self_attn.k_proj", false),
            ("attn_v.weight", "self_attn.v_proj", false),
            ("attn_output.weight", "self_attn.o_proj", false),
        ]
    };

    let mut body = Vec::<u8>::with_capacity(if include_mlp { 140 << 20 } else { 52 << 20 });
    let mut header_entries = Vec::<String>::with_capacity(num_layers * specs.len() * 2);

    let mut attn_cap = 0.0f64;
    let mut attn_tot = 0.0f64;
    let mut mlp_cap = 0.0f64;
    let mut mlp_tot = 0.0f64;

    for l in 0..num_layers {
        for (s_idx, &(gguf_suf, peft_mod, is_mlp)) in specs.iter().enumerate() {
            let tname = format!("blk.{}.{}", l, gguf_suf);
            let (mut w_t, n_t, k_t) = dequant_gguf_2d_f32(&g_targ, &tname)?;
            let (w_b, n_b, k_b) = dequant_gguf_2d_f32(&g_base, &tname)?;
            if n_t != n_b || k_t != k_b {
                return Err(format!(
                    "Shape mismatch on {}: target=[{},{}] vs base=[{},{}]",
                    tname, n_t, k_t, n_b, k_b
                ));
            }
            for i in 0..w_t.len() {
                w_t[i] -= w_b[i];
            }
            let seed = ((l as u64) * 31 + (s_idx as u64) + 1) * 0x9e37_79b9;
            let (lora_a, lora_b, cap_e, tot_e) = randomized_svd_rank_r(&w_t, n_t, k_t, rank, seed);
            if is_mlp {
                mlp_cap += cap_e;
                mlp_tot += tot_e;
            } else {
                attn_cap += cap_e;
                attn_tot += tot_e;
            }

            let a_start = body.len();
            for &v in &lora_a {
                body.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
            }
            let a_end = body.len();
            header_entries.push(format!(
                "\"base_model.model.model.layers.{}.{}.lora_A.weight\":{{\"dtype\":\"F16\",\"shape\":[{},{}],\"data_offsets\":[{},{}]}}",
                l, peft_mod, rank, k_t, a_start, a_end
            ));

            let b_start = body.len();
            for &v in &lora_b {
                body.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
            }
            let b_end = body.len();
            header_entries.push(format!(
                "\"base_model.model.model.layers.{}.{}.lora_B.weight\":{{\"dtype\":\"F16\",\"shape\":[{},{}],\"data_offsets\":[{},{}]}}",
                l, peft_mod, n_t, rank, b_start, b_end
            ));
        }
    }

    let mut hdr_json = format!("{{{}}}", header_entries.join(","));
    while hdr_json.len() % 8 != 0 {
        hdr_json.push(' ');
    }
    let mut file_bytes = Vec::with_capacity(8 + hdr_json.len() + body.len());
    file_bytes.extend_from_slice(&(hdr_json.len() as u64).to_le_bytes());
    file_bytes.extend_from_slice(hdr_json.as_bytes());
    file_bytes.extend_from_slice(&body);

    let st_bytes = file_bytes.len();
    fs::write(out_dir.join("adapter_model.safetensors"), file_bytes).map_err(|e| e.to_string())?;

    let all_cap = attn_cap + mlp_cap;
    let all_tot = attn_tot + mlp_tot;
    Ok(SvdExtractStats {
        num_layers,
        rank,
        include_mlp,
        num_tensors: header_entries.len(),
        safetensors_bytes: st_bytes,
        attn_energy_pct: if attn_tot > 0.0 { 100.0 * attn_cap / attn_tot } else { 0.0 },
        mlp_energy_pct: if mlp_tot > 0.0 { 100.0 * mlp_cap / mlp_tot } else { 0.0 },
        total_energy_pct: if all_tot > 0.0 { 100.0 * all_cap / all_tot } else { 0.0 },
        elapsed_s: t0.elapsed().as_secs_f64(),
    })
}
