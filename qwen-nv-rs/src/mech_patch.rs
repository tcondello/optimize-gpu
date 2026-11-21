//! Zero-Retraining Mechanistic Causal Localization, Surgical `BF16 safetensors` -> `Q8_0 GGUF`
//! Orthogonal Weight Patching ($W_{\text{new}} = (I - \alpha D_l D_l^\top) W$), KL-Divergence Gate,
//! & Instant Tensor-Level Rollback (`Phase 2` – `Phase 4`).
//!
//! Strictly enforces `AGENTS.md`: `0` external Rust crates, `0` PyTorch, `0` Docker.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};
use std::fs::{self, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;
use std::time::Instant;

use crate::gguf::{f16_to_f32, f32_to_bf16_bits, f32_to_f16_bits, resolve_model_path, GgufFile, MmapFile};
use crate::lora::dequant_gguf_2d_f32;
use crate::nv_defs::{get_u32, get_u64};

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

const DIRS_MAGIC: &[u8; 4] = b"QMDR";
const BACKUP_MAGIC: &[u8; 4] = b"QMBK";

// -----------------------------------------------------------------------------
// Zero-Copy `mmap` Safetensors Shard Reader & Index Resolver
// -----------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct StSliceMeta {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub file_data_start: usize, // absolute byte offset inside `.safetensors` file
    pub byte_len: usize,
}

pub struct MmapSafetensors {
    pub mmap: MmapFile,
    pub header_end: usize,
    pub entries: HashMap<String, StSliceMeta>,
}

impl MmapSafetensors {
    pub fn open(path: &Path) -> Result<Self, String> {
        let mmap = MmapFile::open(path)?;
        let raw = mmap.as_slice();
        if raw.len() < 8 {
            return Err(format!("Safetensors {} too small", path.display()));
        }
        let hdr_len = get_u64(raw, 0) as usize;
        if 8 + hdr_len > raw.len() {
            return Err(format!("Invalid header length in {}", path.display()));
        }
        let header_end = 8 + hdr_len;
        let hdr_str = std::str::from_utf8(&raw[8..header_end])
            .map_err(|e| format!("Invalid UTF-8 safetensors header: {}", e))?;
        let entries = parse_st_header(hdr_str, header_end)?;
        Ok(Self {
            mmap,
            header_end,
            entries,
        })
    }

    pub fn decode_f32(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
        let meta = self
            .entries
            .get(name)
            .ok_or_else(|| format!("Tensor '{}' not found in safetensors shard", name))?;
        let s = &self.mmap.as_slice()[meta.file_data_start..meta.file_data_start + meta.byte_len];
        let nel: usize = meta.shape.iter().product();
        let mut out = Vec::with_capacity(nel);
        match meta.dtype.as_str() {
            "BF16" => {
                for c in s.chunks_exact(2) {
                    let b = u16::from_le_bytes([c[0], c[1]]) as u32;
                    out.push(f32::from_bits(b << 16));
                }
            }
            "F16" => {
                for c in s.chunks_exact(2) {
                    out.push(f16_to_f32(u16::from_le_bytes([c[0], c[1]])));
                }
            }
            "F32" => {
                for c in s.chunks_exact(4) {
                    out.push(f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]])));
                }
            }
            other => return Err(format!("Unsupported safetensors dtype '{}'", other)),
        }
        Ok((out, meta.shape.clone()))
    }

    pub fn raw_tensor_bytes(&self, name: &str) -> Result<(&[u8], StSliceMeta), String> {
        let meta = self
            .entries
            .get(name)
            .ok_or_else(|| format!("Tensor '{}' not found in safetensors shard", name))?
            .clone();
        let s = &self.mmap.as_slice()[meta.file_data_start..meta.file_data_start + meta.byte_len];
        Ok((s, meta))
    }
}

fn parse_st_header(hdr: &str, data_base: usize) -> Result<HashMap<String, StSliceMeta>, String> {
    let bytes = hdr.as_bytes();
    let mut i = 0usize;
    let skip_ws = |mut p: usize| -> usize {
        while p < bytes.len() && bytes[p].is_ascii_whitespace() {
            p += 1;
        }
        p
    };
    let read_qstr = |mut p: usize| -> Result<(String, usize), String> {
        p = skip_ws(p);
        if p >= bytes.len() || bytes[p] != b'"' {
            return Err("Expected '\"' in safetensors header".into());
        }
        p += 1;
        let mut s = String::new();
        while p < bytes.len() {
            let c = bytes[p];
            p += 1;
            if c == b'"' {
                return Ok((s, p));
            }
            if c == b'\\' && p < bytes.len() {
                s.push(bytes[p] as char);
                p += 1;
            } else {
                s.push(c as char);
            }
        }
        Err("Unterminated string in safetensors header".into())
    };

    i = skip_ws(i);
    if i >= bytes.len() || bytes[i] != b'{' {
        return Err("Safetensors header missing '{'".into());
    }
    i += 1;

    let mut map = HashMap::new();
    loop {
        i = skip_ws(i);
        if i >= bytes.len() || bytes[i] == b'}' {
            break;
        }
        if bytes[i] == b',' {
            i += 1;
            continue;
        }
        let (key, next_i) = read_qstr(i)?;
        i = skip_ws(next_i);
        if i >= bytes.len() || bytes[i] != b':' {
            return Err("Expected ':' in safetensors header".into());
        }
        i = skip_ws(i + 1);
        if bytes[i] != b'{' {
            return Err("Expected '{' for tensor object".into());
        }
        let mut depth = 1usize;
        let obj_start = i + 1;
        i += 1;
        while i < bytes.len() && depth > 0 {
            match bytes[i] {
                b'"' => {
                    i += 1;
                    while i < bytes.len() {
                        if bytes[i] == b'\\' {
                            i += 2;
                        } else if bytes[i] == b'"' {
                            i += 1;
                            break;
                        } else {
                            i += 1;
                        }
                    }
                }
                b'{' => {
                    depth += 1;
                    i += 1;
                }
                b'}' => {
                    depth -= 1;
                    i += 1;
                }
                _ => i += 1,
            }
        }
        let obj_str = &hdr[obj_start..i - 1];
        if key == "__metadata__" {
            continue;
        }
        let dtype = extract_str_val(obj_str, "\"dtype\"").unwrap_or_default();
        let shape = extract_usize_arr(obj_str, "\"shape\"").unwrap_or_default();
        let offs = extract_usize_arr(obj_str, "\"data_offsets\"").unwrap_or_default();
        if offs.len() == 2 && offs[1] >= offs[0] {
            map.insert(
                key,
                StSliceMeta {
                    dtype,
                    shape,
                    file_data_start: data_base + offs[0],
                    byte_len: offs[1] - offs[0],
                },
            );
        }
    }
    Ok(map)
}

fn extract_str_val(src: &str, key: &str) -> Option<String> {
    let idx = src.find(key)?;
    let rest = src[idx + key.len()..].trim_start().strip_prefix(':')?.trim_start().strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

fn extract_usize_arr(src: &str, key: &str) -> Option<Vec<usize>> {
    let idx = src.find(key)?;
    let rest = src[idx + key.len()..].trim_start().strip_prefix(':')?.trim_start().strip_prefix('[')?;
    let end = rest.find(']')?;
    let inner = rest[..end].trim();
    if inner.is_empty() {
        return Some(Vec::new());
    }
    let mut out = Vec::new();
    for p in inner.split(',') {
        out.push(p.trim().parse::<usize>().ok()?);
    }
    Some(out)
}

/// Parse `model.safetensors.index.json` -> `HashMap<tensor_name, shard_filename>`.
pub fn parse_safetensors_index(dir: &Path) -> Result<HashMap<String, String>, String> {
    let idx_path = dir.join("model.safetensors.index.json");
    if !idx_path.exists() {
        let single = dir.join("model.safetensors");
        if single.exists() {
            let st = MmapSafetensors::open(&single)?;
            let mut m = HashMap::new();
            for k in st.entries.keys() {
                m.insert(k.clone(), "model.safetensors".to_string());
            }
            return Ok(m);
        }
        return Err(format!("Missing model.safetensors.index.json in {}", dir.display()));
    }
    let txt = fs::read_to_string(&idx_path).map_err(|e| e.to_string())?;
    let wm_idx = txt
        .find("\"weight_map\"")
        .ok_or("Missing weight_map in model.safetensors.index.json")?;
    let body = &txt[wm_idx + 12..];
    let open_brace = body.find('{').ok_or("Malformed weight_map")?;
    let sub = &body[open_brace + 1..];

    let mut map = HashMap::new();
    for line in sub.lines() {
        let ln = line.trim().trim_end_matches(',');
        if ln.starts_with('}') {
            break;
        }
        if let Some(colon) = ln.find(':') {
            let k = ln[..colon].trim().trim_matches('"');
            let v = ln[colon + 1..].trim().trim_matches('"');
            if !k.is_empty() && v.ends_with(".safetensors") {
                map.insert(k.to_string(), v.to_string());
            }
        }
    }
    Ok(map)
}

// -----------------------------------------------------------------------------
// Phase 2: Causal Localization (`difference-in-means` + Top-2 SVD + Separability)
// -----------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct LayerCausalStats {
    pub layer: usize,
    pub res_cos_sep: f64,
    pub res_cohen_d: f64,
    pub o_proj_cos_sep: f64,
    pub down_proj_cos_sep: f64,
    pub o_proj_contrib: f64,
    pub down_proj_contrib: f64,
    pub combined_score: f64,
}

/// Analyze GPU residual-stream hook dumps (`[n_samples, 1 + 2*num_layers, dim]` raw `f32`)
/// for `failure_set` vs `clean_set`, extracting `k=2` orthonormal failure directions per
/// `(layer, writer)` (`writer 0 = o_proj / h_mid`, `writer 1 = down_proj / h_out`),
/// saving binary directions to `out_dirs_bin` and JSON rankings to `out_report_json`.
pub fn localize_causal_directions(
    fail_hooks_bin: &Path,
    clean_hooks_bin: &Path,
    num_layers: usize,
    dim: usize,
    out_dirs_bin: &Path,
    out_report_json: &Path,
) -> Result<Vec<LayerCausalStats>, String> {
    let slots = 1 + 2 * num_layers;
    let stride = slots * dim;

    let read_hook_file = |p: &Path| -> Result<(Vec<f32>, usize), String> {
        let raw = fs::read(p).map_err(|e| format!("Failed reading {}: {}", p.display(), e))?;
        if raw.len() % (stride * 4) != 0 || raw.is_empty() {
            return Err(format!(
                "Hook file {} byte len {} is not a non-zero multiple of stride*4 ({})",
                p.display(),
                raw.len(),
                stride * 4
            ));
        }
        let n = raw.len() / (stride * 4);
        let vals: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_bits(get_u32(c, 0)))
            .collect();
        Ok((vals, n))
    };

    let (fail_vals, n_fail) = read_hook_file(fail_hooks_bin)?;
    let (clean_vals, n_clean) = read_hook_file(clean_hooks_bin)?;

    let normalize_vec = |v: &mut [f64]| -> f64 {
        let norm = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm > 1e-12 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
        norm
    };

    let dot_f32_f64 = |x: &[f32], d: &[f64]| -> f64 {
        let mut s = 0.0f64;
        for i in 0..d.len() {
            s += (x[i] as f64) * d[i];
        }
        s
    };

    let cos_f32_f64 = |x: &[f32], d: &[f64]| -> f64 {
        let mut s = 0.0f64;
        let mut nx = 0.0f64;
        for i in 0..d.len() {
            let xi = x[i] as f64;
            s += xi * d[i];
            nx += xi * xi;
        }
        if nx > 1e-20 {
            s / nx.sqrt()
        } else {
            0.0
        }
    };

    // Extract top-2 orthonormal directions `[d0, d1]`:
    // `d0` = normalized difference-in-means `mu_fail - mu_clean`
    // `d1` = 1st singular vector of `(x_fail - mu_clean)` projected orthogonal to `d0` (power iteration)
    let compute_k2_dirs = |slot_idx: usize| -> ([Vec<f64>; 2], f64, f64) {
        let mut mu_f = vec![0.0f64; dim];
        let mut mu_c = vec![0.0f64; dim];
        for s in 0..n_fail {
            let row = &fail_vals[s * stride + slot_idx * dim..s * stride + (slot_idx + 1) * dim];
            for i in 0..dim {
                mu_f[i] += (row[i] as f64) / (n_fail as f64);
            }
        }
        for s in 0..n_clean {
            let row = &clean_vals[s * stride + slot_idx * dim..s * stride + (slot_idx + 1) * dim];
            for i in 0..dim {
                mu_c[i] += (row[i] as f64) / (n_clean as f64);
            }
        }
        let mut d0: Vec<f64> = (0..dim).map(|i| mu_f[i] - mu_c[i]).collect();
        normalize_vec(&mut d0);

        // Cosine separability & Cohen's d along `d0`
        let mut f_cos = 0.0f64;
        let mut c_cos = 0.0f64;
        let mut f_proj = Vec::with_capacity(n_fail);
        let mut c_proj = Vec::with_capacity(n_clean);
        for s in 0..n_fail {
            let row = &fail_vals[s * stride + slot_idx * dim..s * stride + (slot_idx + 1) * dim];
            f_cos += cos_f32_f64(row, &d0) / (n_fail as f64);
            f_proj.push(dot_f32_f64(row, &d0));
        }
        for s in 0..n_clean {
            let row = &clean_vals[s * stride + slot_idx * dim..s * stride + (slot_idx + 1) * dim];
            c_cos += cos_f32_f64(row, &d0) / (n_clean as f64);
            c_proj.push(dot_f32_f64(row, &d0));
        }
        let mf: f64 = f_proj.iter().sum::<f64>() / (n_fail as f64);
        let mc: f64 = c_proj.iter().sum::<f64>() / (n_clean as f64);
        let vf: f64 = f_proj.iter().map(|&v| (v - mf) * (v - mf)).sum::<f64>() / (n_fail.max(1) as f64);
        let vc: f64 = c_proj.iter().map(|&v| (v - mc) * (v - mc)).sum::<f64>() / (n_clean.max(1) as f64);
        let pooled_sd = (0.5 * (vf + vc)).sqrt().max(1e-9);
        let cohen_d = (mf - mc) / pooled_sd;

        // Orthogonal 2nd SVD component `d1` on residual `(x_fail - mu_c) - proj_{d0}`
        let mut d1: Vec<f64> = (0..dim)
            .map(|i| (((i * 1315423911) ^ 0x5a5a5a5a) as f64) * 1e-9 - 0.5)
            .collect();
        let dot01: f64 = d1.iter().zip(d0.iter()).map(|(a, b)| a * b).sum();
        for i in 0..dim {
            d1[i] -= dot01 * d0[i];
        }
        normalize_vec(&mut d1);

        for _iter in 0..20 {
            let mut next_d = vec![0.0f64; dim];
            for s in 0..n_fail {
                let row = &fail_vals[s * stride + slot_idx * dim..s * stride + (slot_idx + 1) * dim];
                let mut proj1 = 0.0f64;
                let mut proj0 = 0.0f64;
                for i in 0..dim {
                    let diff = (row[i] as f64) - mu_c[i];
                    proj1 += diff * d1[i];
                    proj0 += diff * d0[i];
                }
                for i in 0..dim {
                    let diff_orth = ((row[i] as f64) - mu_c[i]) - proj0 * d0[i];
                    next_d[i] += proj1 * diff_orth;
                }
            }
            let d0_ov: f64 = next_d.iter().zip(d0.iter()).map(|(a, b)| a * b).sum();
            for i in 0..dim {
                next_d[i] -= d0_ov * d0[i];
            }
            if normalize_vec(&mut next_d) < 1e-12 {
                break;
            }
            d1 = next_d;
        }

        ([d0, d1], f_cos - c_cos, cohen_d)
    };

    // Binary layout of `out_dirs_bin`:
    // Header: `QMDR` (4B) + ver=1 (u32) + num_layers (u32) + dim (u32)
    // Body:   for layer in 0..num_layers:
    //           writer 0 (`o_proj`, derived from post-attn `h_mid` & writer delta): `[2, dim]` f32
    //           writer 1 (`down_proj`, derived from post-mlp `h_out` & writer delta): `[2, dim]` f32
    let mut dirs_bytes = Vec::with_capacity(16 + num_layers * 2 * 2 * dim * 4);
    dirs_bytes.extend_from_slice(DIRS_MAGIC);
    dirs_bytes.extend_from_slice(&1u32.to_le_bytes());
    dirs_bytes.extend_from_slice(&(num_layers as u32).to_le_bytes());
    dirs_bytes.extend_from_slice(&(dim as u32).to_le_bytes());

    let mut stats_list = Vec::with_capacity(num_layers);

    for l in 0..num_layers {
        let slot_in = 2 * l;
        let slot_mid = 2 * l + 1;
        let slot_out = 2 * l + 2;

        let (dirs_mid, mid_cos_sep, _) = compute_k2_dirs(slot_mid);
        let (dirs_out, res_cos_sep, res_cohen_d) = compute_k2_dirs(slot_out);

        // Compute direct writer causal injection `(mu_writer_fail - mu_writer_clean) · d_out`
        // AND writer-level cosine separability!
        let mut o_diff = vec![0.0f64; dim];
        let mut dn_diff = vec![0.0f64; dim];
        for s in 0..n_fail {
            let base = s * stride;
            for i in 0..dim {
                let w_o = (fail_vals[base + slot_mid * dim + i] - fail_vals[base + slot_in * dim + i]) as f64;
                let w_dn = (fail_vals[base + slot_out * dim + i] - fail_vals[base + slot_mid * dim + i]) as f64;
                o_diff[i] += w_o / (n_fail as f64);
                dn_diff[i] += w_dn / (n_fail as f64);
            }
        }
        for s in 0..n_clean {
            let base = s * stride;
            for i in 0..dim {
                let w_o = (clean_vals[base + slot_mid * dim + i] - clean_vals[base + slot_in * dim + i]) as f64;
                let w_dn = (clean_vals[base + slot_out * dim + i] - clean_vals[base + slot_mid * dim + i]) as f64;
                o_diff[i] -= w_o / (n_clean as f64);
                dn_diff[i] -= w_dn / (n_clean as f64);
            }
        }

        let o_contrib: f64 = o_diff.iter().zip(dirs_out[0].iter()).map(|(a, b)| a * b).sum();
        let dn_contrib: f64 = dn_diff.iter().zip(dirs_out[0].iter()).map(|(a, b)| a * b).sum();

        // Writer-specific normalized directions for surgical projection:
        // Notice: `h_l_out` accumulates across layers, so projecting `o_proj` and `down_proj` orthogonal
        // to the residual failure direction `dirs_out` prevents writer `l` from writing along `d_l`!
        for k_idx in 0..2 {
            for &v in &dirs_mid[k_idx] {
                dirs_bytes.extend_from_slice(&(v as f32).to_bits().to_le_bytes());
            }
        }
        for k_idx in 0..2 {
            for &v in &dirs_out[k_idx] {
                dirs_bytes.extend_from_slice(&(v as f32).to_bits().to_le_bytes());
            }
        }

        let combined = res_cos_sep * 0.5 + mid_cos_sep * 0.2 + 0.3 * (res_cohen_d.max(0.0) / 4.0);
        stats_list.push(LayerCausalStats {
            layer: l,
            res_cos_sep,
            res_cohen_d,
            o_proj_cos_sep: mid_cos_sep,
            down_proj_cos_sep: res_cos_sep,
            o_proj_contrib: o_contrib,
            down_proj_contrib: dn_contrib,
            combined_score: combined,
        });
    }

    if let Some(p) = out_dirs_bin.parent() {
        if !p.as_os_str().is_empty() {
            fs::create_dir_all(p).map_err(|e| e.to_string())?;
        }
    }
    fs::write(out_dirs_bin, dirs_bytes).map_err(|e| e.to_string())?;

    let mut ranked = stats_list.clone();
    ranked.sort_by(|a, b| {
        b.combined_score
            .partial_cmp(&a.combined_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let layers_json: Vec<String> = ranked
        .iter()
        .map(|s| {
            format!(
                "    {{\"layer\":{},\"combined_score\":{:.6},\"res_cos_sep\":{:.6},\"res_cohen_d\":{:.4},\"o_proj_cos_sep\":{:.6},\"o_proj_contrib\":{:.4},\"down_proj_contrib\":{:.4}}}",
                s.layer,
                s.combined_score,
                s.res_cos_sep,
                s.res_cohen_d,
                s.o_proj_cos_sep,
                s.o_proj_contrib,
                s.down_proj_contrib
            )
        })
        .collect();
    let report = format!(
        "{{\n  \"n_fail\": {},\n  \"n_clean\": {},\n  \"num_layers\": {},\n  \"dim\": {},\n  \"ranked_layers\": [\n{}\n  ]\n}}\n",
        n_fail,
        n_clean,
        num_layers,
        dim,
        layers_json.join(",\n")
    );
    if let Some(p) = out_report_json.parent() {
        if !p.as_os_str().is_empty() {
            fs::create_dir_all(p).map_err(|e| e.to_string())?;
        }
    }
    fs::write(out_report_json, report).map_err(|e| e.to_string())?;

    Ok(ranked)
}

fn load_dirs_bin(dirs_bin: &Path) -> Result<(usize, usize, Vec<f32>), String> {
    let raw = fs::read(dirs_bin).map_err(|e| format!("Failed reading {}: {}", dirs_bin.display(), e))?;
    if raw.len() < 16 || &raw[0..4] != DIRS_MAGIC {
        return Err(format!("Invalid QMDR header in {}", dirs_bin.display()));
    }
    let num_layers = get_u32(&raw, 8) as usize;
    let dim = get_u32(&raw, 12) as usize;
    let need = 16 + num_layers * 4 * dim * 4;
    if raw.len() != need {
        return Err(format!("QMDR size mismatch ({} != {})", raw.len(), need));
    }
    let vals: Vec<f32> = raw[16..]
        .chunks_exact(4)
        .map(|c| f32::from_bits(get_u32(c, 0)))
        .collect();
    Ok((num_layers, dim, vals))
}

// -----------------------------------------------------------------------------
// Phase 2.4: Fast Runtime Hook-Equivalent Rank-k PEFT Adapter Builder
// (`(I - alpha D D^T) W = W + (-alpha D)(D^T W)` from `BF16` master shards!)
// -----------------------------------------------------------------------------

pub fn build_ablation_lora_from_bf16(
    bf16_dir: &Path,
    dirs_bin: &Path,
    layers: &[usize],
    edit_o_proj: bool,
    edit_down_proj: bool,
    rank_k: usize,
    alpha: f32,
    out_lora_dir: &Path,
) -> Result<(), String> {
    if rank_k == 0 || rank_k > 2 {
        return Err("rank_k must be 1 or 2".into());
    }
    let (num_layers, dim, dirs_vals) = load_dirs_bin(dirs_bin)?;
    let is_gguf_master = bf16_dir.is_file()
        || bf16_dir.extension().and_then(|s| s.to_str()) == Some("gguf")
        || !bf16_dir.exists();
    let gguf_opt = if is_gguf_master {
        let gp = if bf16_dir.exists() {
            bf16_dir.to_path_buf()
        } else {
            resolve_model_path(&bf16_dir.to_string_lossy())?
        };
        Some(GgufFile::open(&gp)?)
    } else {
        None
    };
    let weight_map = if gguf_opt.is_none() {
        parse_safetensors_index(bf16_dir)?
    } else {
        HashMap::new()
    };
    let mut shard_cache: HashMap<String, MmapSafetensors> = HashMap::new();

    fs::create_dir_all(out_lora_dir).map_err(|e| e.to_string())?;
    let cfg_json = format!(
        "{{\n  \"peft_type\": \"LORA\",\n  \"r\": {},\n  \"lora_alpha\": {:.1},\n  \"target_modules\": [\"o_proj\", \"down_proj\"]\n}}\n",
        rank_k, rank_k as f32
    );
    fs::write(out_lora_dir.join("adapter_config.json"), cfg_json).map_err(|e| e.to_string())?;

    let mut body = Vec::<u8>::new();
    let mut header_entries = Vec::<String>::new();

    for &l in layers {
        if l >= num_layers {
            continue;
        }
        for (writer_idx, hf_mod, gguf_mod, do_it) in [
            (0usize, "self_attn.o_proj", "attn_output", edit_o_proj),
            (1usize, "mlp.down_proj", "ffn_down", edit_down_proj),
        ] {
            if !do_it {
                continue;
            }
            let st_key = format!("model.layers.{}.{}.weight", l, hf_mod);
            let (w_f32, n_rows, k_cols) = if let Some(ref gg) = gguf_opt {
                let gkey = format!("blk.{}.{}.weight", l, gguf_mod);
                dequant_gguf_2d_f32(gg, &gkey)?
            } else {
                let shard_name = weight_map
                    .get(&st_key)
                    .ok_or_else(|| format!("Key {} missing from weight_map", st_key))?;
                if !shard_cache.contains_key(shard_name) {
                    let st = MmapSafetensors::open(&bf16_dir.join(shard_name))?;
                    shard_cache.insert(shard_name.clone(), st);
                }
                let (wf, shape) = shard_cache[shard_name].decode_f32(&st_key)?;
                (wf, shape[0], shape[1])
            };
            if n_rows != dim {
                return Err(format!("Expected n_rows={} on {}, got {}", dim, st_key, n_rows));
            }

            // `d_mat` `[rank_k, dim]`
            let d_base = l * (4 * dim) + writer_idx * (2 * dim);
            let d_mat = &dirs_vals[d_base..d_base + rank_k * dim];

            // `lora_a = D^T W` -> `[rank_k, k_cols]` via Accelerate `cblas_sgemm`
            let mut lora_a = vec![0.0f32; rank_k * k_cols];
            unsafe {
                cblas_sgemm(
                    CBLAS_ROW_MAJOR,
                    CBLAS_NO_TRANS,
                    CBLAS_NO_TRANS,
                    rank_k as i32,
                    k_cols as i32,
                    dim as i32,
                    1.0,
                    d_mat.as_ptr(),
                    dim as i32,
                    w_f32.as_ptr(),
                    k_cols as i32,
                    0.0,
                    lora_a.as_mut_ptr(),
                    k_cols as i32,
                );
            }

            // Balance `sqrt(alpha)` across A and B so FP16 dynamic range stays crisp!
            let sq = alpha.abs().sqrt().max(1e-6);
            let sign = if alpha >= 0.0 { -1.0f32 } else { 1.0f32 };

            // `lora_b` `[dim, rank_k]` = `sign * sq * D^T` (each column `r` is `d_mat[r, :]`)
            let mut lora_b = vec![0.0f32; dim * rank_k];
            for row in 0..dim {
                for r in 0..rank_k {
                    lora_b[row * rank_k + r] = sign * sq * d_mat[r * dim + row];
                }
            }

            let a_start = body.len();
            for &v in &lora_a {
                body.extend_from_slice(&f32_to_f16_bits(v * sq).to_le_bytes());
            }
            let a_end = body.len();
            header_entries.push(format!(
                "\"base_model.model.model.layers.{}.{}.lora_A.weight\":{{\"dtype\":\"F16\",\"shape\":[{},{}],\"data_offsets\":[{},{}]}}",
                l, hf_mod, rank_k, k_cols, a_start, a_end
            ));

            let b_start = body.len();
            for &v in &lora_b {
                body.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
            }
            let b_end = body.len();
            header_entries.push(format!(
                "\"base_model.model.model.layers.{}.{}.lora_B.weight\":{{\"dtype\":\"F16\",\"shape\":[{},{}],\"data_offsets\":[{},{}]}}",
                l, hf_mod, dim, rank_k, b_start, b_end
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
    fs::write(out_lora_dir.join("adapter_model.safetensors"), file_bytes).map_err(|e| e.to_string())
}

// -----------------------------------------------------------------------------
// Phase 3 & Phase 4: Surgical `BF16 safetensors` -> `Q8_0 GGUF` Patcher & Rollback
// -----------------------------------------------------------------------------

/// Symmetric `Q8_0` block encoder (`32` `f32` -> `34` bytes: `f16` scale + `32` `i8`).
pub fn quantize_f32_to_q8_0(vals: &[f32]) -> Vec<u8> {
    assert_eq!(vals.len() % 32, 0);
    let n_blk = vals.len() / 32;
    let mut out = vec![0u8; n_blk * 34];
    for b in 0..n_blk {
        let src = &vals[b * 32..(b + 1) * 32];
        let dst = &mut out[b * 34..(b + 1) * 34];
        let mut amax = 0.0f32;
        for &v in src {
            let av = v.abs();
            if av > amax {
                amax = av;
            }
        }
        let d = amax / 127.0f32;
        let d_f16 = f32_to_f16_bits(d);
        let d_rt = f16_to_f32(d_f16);
        let id = if d_rt > 0.0 { 1.0f32 / d_rt } else { 0.0f32 };
        dst[0..2].copy_from_slice(&d_f16.to_le_bytes());
        for j in 0..32 {
            let q = (src[j] * id).round().clamp(-127.0, 127.0) as i8;
            dst[2 + j] = q as u8;
        }
    }
    out
}

/// Symmetric `Q6_K` super-block encoder (`256` `f32` -> `210` bytes: `128B` `ql`, `64B` `qh`, `16B` `i8` scales, `2B` `f16` `d`).
pub fn quantize_f32_to_q6_k(vals: &[f32]) -> Vec<u8> {
    assert_eq!(vals.len() % 256, 0);
    let n_sb = vals.len() / 256;
    let mut out = vec![0u8; n_sb * 210];
    for sb in 0..n_sb {
        let src = &vals[sb * 256..(sb + 1) * 256];
        let blk = &mut out[sb * 210..(sb + 1) * 210];
        let mut sub_scales = [0.0f32; 16];
        let mut max_ls = 0.0f32;
        for s in 0..16 {
            let mut amax = 0.0f32;
            for j in 0..16 {
                let av = src[s * 16 + j].abs();
                if av > amax {
                    amax = av;
                }
            }
            let ls = amax / 31.0f32;
            sub_scales[s] = ls;
            if ls > max_ls {
                max_ls = ls;
            }
        }
        let d_f16 = f32_to_f16_bits(max_ls / 127.0f32);
        let d_rt = f16_to_f32(d_f16);
        blk[208..210].copy_from_slice(&d_f16.to_le_bytes());

        let mut sc_i8 = [0i8; 16];
        for s in 0..16 {
            sc_i8[s] = if d_rt > 0.0 {
                let q = (sub_scales[s] / d_rt).round().clamp(-127.0, 127.0) as i8;
                if q == 0 && sub_scales[s] > 0.0 { 1 } else { q }
            } else {
                0
            };
            blk[192 + s] = sc_i8[s] as u8;
        }

        for n128 in 0..2usize {
            let src_p = &src[n128 * 128..(n128 + 1) * 128];
            let sc_p = &sc_i8[n128 * 8..(n128 + 1) * 8];
            for l in 0..32usize {
                let is = l / 16;
                let quant_elem = |v: f32, sc_val: i8| -> u8 {
                    let step = d_rt * (sc_val as f32);
                    if step == 0.0 {
                        32u8
                    } else {
                        ((v / step).round() as i32 + 32).clamp(0, 63) as u8
                    }
                };
                let q1 = quant_elem(src_p[l], sc_p[is]);
                let q2 = quant_elem(src_p[l + 32], sc_p[is + 2]);
                let q3 = quant_elem(src_p[l + 64], sc_p[is + 4]);
                let q4 = quant_elem(src_p[l + 96], sc_p[is + 6]);
                blk[n128 * 64 + l] = (q1 & 0x0f) | ((q3 & 0x0f) << 4);
                blk[n128 * 64 + l + 32] = (q2 & 0x0f) | ((q4 & 0x0f) << 4);
                blk[128 + n128 * 32 + l] =
                    ((q1 >> 4) & 3) | (((q2 >> 4) & 3) << 2) | (((q3 >> 4) & 3) << 4) | (((q4 >> 4) & 3) << 6);
            }
        }
    }
    out
}

#[derive(Clone, Debug)]
pub struct PatchSummary {
    pub out_dir: String,
    pub gguf_path: String,
    pub edited_tensors: Vec<String>,
    pub layers: Vec<usize>,
    pub rank_k: usize,
    pub alpha: f32,
    pub rescale_row_norms: bool,
    pub subspace_dof_params: usize,
    pub subspace_dof_pct: f64,
    pub mean_rel_frobenius_delta: f64,
    pub elapsed_s: f64,
}

/// Apply surgical orthogonal weight projection $W_{\text{new}} = (I - \alpha D_l D_l^\top) W$
/// directly to the `BF16` master `safetensors` shards, save backup records for `<50 ms` rollback,
/// and update the corresponding `Q8_0` matrices inside `Qwen3-4B-Thinking-2507-Patched-Q8_0.gguf`.
pub fn apply_surgical_weight_patch(
    master_bf16_dir: &Path,
    base_q8_gguf: &Path,
    dirs_bin: &Path,
    out_ckpt_dir: &Path,
    layers: &[usize],
    edit_o_proj: bool,
    edit_down_proj: bool,
    rank_k: usize,
    alpha: f32,
    rescale_row_norms: bool,
) -> Result<PatchSummary, String> {
    let t0 = Instant::now();
    if rank_k == 0 || rank_k > 2 {
        return Err("rank_k must be 1 or 2".into());
    }

    fs::create_dir_all(out_ckpt_dir).map_err(|e| e.to_string())?;
    let backup_path = out_ckpt_dir.join("rollback_backup.bin");
    if backup_path.exists() {
        // Always restore clean baseline bytes before applying a new candidate patch!
        let _ = rollback_surgical_weight_patch(out_ckpt_dir)?;
    }

    let is_gguf_only = master_bf16_dir.is_file()
        || master_bf16_dir.extension().and_then(|s| s.to_str()) == Some("gguf")
        || !master_bf16_dir.exists()
        || master_bf16_dir == base_q8_gguf;

    let weight_map = if !is_gguf_only {
        // Copy metadata files into `out_ckpt_dir`
        for meta_f in [
            "config.json",
            "generation_config.json",
            "tokenizer_config.json",
            "model.safetensors.index.json",
        ] {
            let src = master_bf16_dir.join(meta_f);
            let dst = out_ckpt_dir.join(meta_f);
            if src.exists() && !dst.exists() {
                fs::copy(&src, &dst).map_err(|e| e.to_string())?;
            }
        }

        let wm = parse_safetensors_index(master_bf16_dir)?;
        let all_shards: HashSet<String> = wm.values().cloned().collect();

        // Identify which `.safetensors` shards will actually be modified
        let mut touched_shards = HashSet::<String>::new();
        for &l in layers {
            if edit_o_proj {
                let k = format!("model.layers.{}.self_attn.o_proj.weight", l);
                if let Some(sh) = wm.get(&k) {
                    touched_shards.insert(sh.clone());
                }
            }
            if edit_down_proj {
                let k = format!("model.layers.{}.mlp.down_proj.weight", l);
                if let Some(sh) = wm.get(&k) {
                    touched_shards.insert(sh.clone());
                }
            }
        }

        // Hardlink untouched shards (`0` extra disk bytes!) and make independent writable copies ONLY for touched shards!
        for sh in &all_shards {
            let src = master_bf16_dir.join(sh);
            let dst = out_ckpt_dir.join(sh);
            if touched_shards.contains(sh) {
                let need_copy = if !dst.exists() {
                    true
                } else {
                    use std::os::unix::fs::MetadataExt;
                    fs::metadata(&dst).map(|m| m.nlink() > 1).unwrap_or(true)
                };
                if need_copy {
                    if dst.exists() {
                        let _ = fs::remove_file(&dst);
                    }
                    eprintln!("[mech-patch] Creating writable BF16 shard copy {}...", dst.display());
                    fs::copy(&src, &dst).map_err(|e| e.to_string())?;
                }
            } else if !dst.exists() {
                if fs::hard_link(&src, &dst).is_err() {
                    fs::copy(&src, &dst).map_err(|e| e.to_string())?;
                }
            }
        }
        wm
    } else {
        HashMap::new()
    };

    let (num_layers, dim, dirs_vals) = load_dirs_bin(dirs_bin)?;
    let base_gguf = GgufFile::open(base_q8_gguf)?;
    let patched_gguf_fname = if !is_gguf_only {
        "Qwen3-4B-Thinking-2507-Patched-Q8_0.gguf".to_string()
    } else if dim == 4096 {
        let sample_t = base_gguf
            .tensors
            .get("blk.0.ffn_down.weight")
            .map(|t| t.ggml_type)
            .unwrap_or(14);
        if sample_t == 14 {
            "Qwen3-8B-Patched-Q6_K.gguf".to_string()
        } else {
            "Qwen3-8B-Patched-Q8_0.gguf".to_string()
        }
    } else {
        let stem = base_q8_gguf
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model");
        format!("{}-Patched.gguf", stem)
    };
    let patched_gguf_path = out_ckpt_dir.join(&patched_gguf_fname);
    if !patched_gguf_path.exists() {
        eprintln!(
            "[mech-patch] Initializing writable GGUF {}...",
            patched_gguf_path.display()
        );
        fs::copy(base_q8_gguf, &patched_gguf_path).map_err(|e| e.to_string())?;
    }

    let mut master_shard_cache: HashMap<String, MmapSafetensors> = HashMap::new();

    // Backup file format (`QMBK`):
    // `QMBK` (4B) + ver=1 (u32) + num_records (u32)
    // Each record: `kind` (u8: 1=safetensors shard, 2=GGUF) + rel_filename_len (u32) + rel_filename
    //              + file_offset (u64) + data_len (u64) + raw original bytes!
    let mut backup_records = Vec::<(u8, String, u64, Vec<u8>)>::new();
    let mut edited_keys = Vec::<String>::new();
    let mut rel_deltas = Vec::<f64>::new();
    let mut subspace_dof = 0usize;

    for &l in layers {
        if l >= num_layers {
            continue;
        }
        for (writer_idx, hf_mod, gguf_mod, do_it) in [
            (0usize, "self_attn.o_proj", "attn_output", edit_o_proj),
            (1usize, "mlp.down_proj", "ffn_down", edit_down_proj),
        ] {
            if !do_it {
                continue;
            }
            let st_key = format!("model.layers.{}.{}.weight", l, hf_mod);
            let gguf_key = format!("blk.{}.{}.weight", l, gguf_mod);

            let (mut w_f32, orig_f32, n_rows, k_cols, st_patch_opt) = if !is_gguf_only {
                let shard_name = weight_map
                    .get(&st_key)
                    .ok_or_else(|| format!("Missing {} in weight_map", st_key))?
                    .clone();
                if !master_shard_cache.contains_key(&shard_name) {
                    let st = MmapSafetensors::open(&master_bf16_dir.join(&shard_name))?;
                    master_shard_cache.insert(shard_name.clone(), st);
                }

                let (orig_st_bytes, st_meta) = master_shard_cache[&shard_name].raw_tensor_bytes(&st_key)?;
                backup_records.push((
                    1u8,
                    shard_name.clone(),
                    st_meta.file_data_start as u64,
                    orig_st_bytes.to_vec(),
                ));

                let (wf, shape) = master_shard_cache[&shard_name].decode_f32(&st_key)?;
                let orig_copy = wf.clone();
                (wf, orig_copy, shape[0], shape[1], Some((shard_name, st_meta)))
            } else {
                let (wf, nr, kc) = dequant_gguf_2d_f32(&base_gguf, &gguf_key)?;
                let orig_copy = wf.clone();
                (wf, orig_copy, nr, kc, None)
            };
            if n_rows != dim {
                return Err(format!("Expected n_rows={} on {}, got {}", dim, st_key, n_rows));
            }

            let orig_row_norms: Vec<f64> = if rescale_row_norms {
                (0..n_rows)
                    .map(|r| {
                        let row = &w_f32[r * k_cols..(r + 1) * k_cols];
                        row.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt()
                    })
                    .collect()
            } else {
                Vec::new()
            };
            let orig_frob_sq: f64 = w_f32.iter().map(|&v| (v as f64) * (v as f64)).sum();

            // Orthogonal ablation: `W_new = W - alpha * D * (D^T * W)`
            let d_base = l * (4 * dim) + writer_idx * (2 * dim);
            let d_mat = &dirs_vals[d_base..d_base + rank_k * dim]; // `[rank_k, dim]`
            let mut dt_w = vec![0.0f32; rank_k * k_cols];
            unsafe {
                cblas_sgemm(
                    CBLAS_ROW_MAJOR,
                    CBLAS_NO_TRANS,
                    CBLAS_NO_TRANS,
                    rank_k as i32,
                    k_cols as i32,
                    dim as i32,
                    1.0,
                    d_mat.as_ptr(),
                    dim as i32,
                    w_f32.as_ptr(),
                    k_cols as i32,
                    0.0,
                    dt_w.as_mut_ptr(),
                    k_cols as i32,
                );
                cblas_sgemm(
                    CBLAS_ROW_MAJOR,
                    CBLAS_TRANS,
                    CBLAS_NO_TRANS,
                    dim as i32,
                    k_cols as i32,
                    rank_k as i32,
                    -alpha,
                    d_mat.as_ptr(),
                    dim as i32,
                    dt_w.as_ptr(),
                    k_cols as i32,
                    1.0,
                    w_f32.as_mut_ptr(),
                    k_cols as i32,
                );
            }

            if rescale_row_norms {
                for r in 0..n_rows {
                    let row = &mut w_f32[r * k_cols..(r + 1) * k_cols];
                    let new_n = row.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt();
                    if new_n > 1e-12 && orig_row_norms[r] > 1e-12 {
                        let scale = (orig_row_norms[r] / new_n) as f32;
                        for x in row.iter_mut() {
                            *x *= scale;
                        }
                    }
                }
            }

            let diff_sq: f64 = w_f32
                .iter()
                .zip(orig_f32.iter())
                .map(|(&a, &b)| {
                    let d = (a - b) as f64;
                    d * d
                })
                .sum();
            rel_deltas.push((diff_sq / orig_frob_sq.max(1e-20)).sqrt());

            // 1. Optional: Write patched `BF16` bytes into `out_ckpt_dir/<shard_name>`
            if let Some((shard_name, st_meta)) = st_patch_opt {
                let mut new_bf16_bytes = Vec::with_capacity(w_f32.len() * 2);
                for &v in &w_f32 {
                    new_bf16_bytes.extend_from_slice(&f32_to_bf16_bits(v).to_le_bytes());
                }
                if new_bf16_bytes.len() != st_meta.byte_len {
                    return Err(format!("BF16 byte len mismatch on {}", st_key));
                }
                let mut f_sh = OpenOptions::new()
                    .write(true)
                    .open(out_ckpt_dir.join(&shard_name))
                    .map_err(|e| e.to_string())?;
                f_sh.seek(SeekFrom::Start(st_meta.file_data_start as u64))
                    .map_err(|e| e.to_string())?;
                f_sh.write_all(&new_bf16_bytes).map_err(|e| e.to_string())?;
            }

            // 2. Quantize patched `FP32` matrix to `Q8_0` or `Q6_K` matching `base_gguf` and patch in-place!
            let g_info = base_gguf
                .tensors
                .get(&gguf_key)
                .ok_or_else(|| format!("Missing {} in base GGUF", gguf_key))?;
            let q_bytes = match g_info.ggml_type {
                8 => quantize_f32_to_q8_0(&w_f32),
                14 => quantize_f32_to_q6_k(&w_f32),
                other => {
                    return Err(format!(
                        "Expected Q8_0 (type=8) or Q6_K (type=14) on {} in base GGUF, got type={}",
                        gguf_key, other
                    ))
                }
            };
            let g_file_off = (base_gguf.data_start + g_info.offset) as u64;
            let orig_g_bytes = base_gguf.tensor_bytes(g_info, q_bytes.len());
            backup_records.push((
                2u8,
                patched_gguf_fname.clone(),
                g_file_off,
                orig_g_bytes.to_vec(),
            ));
            {
                let mut f_gg = OpenOptions::new()
                    .write(true)
                    .open(&patched_gguf_path)
                    .map_err(|e| e.to_string())?;
                f_gg.seek(SeekFrom::Start(g_file_off)).map_err(|e| e.to_string())?;
                f_gg.write_all(&q_bytes).map_err(|e| e.to_string())?;
            }

            edited_keys.push(if is_gguf_only { gguf_key } else { st_key });
            subspace_dof += rank_k * dim;
        }
    }

    // Write `rollback_backup.bin`
    let mut bk_buf = Vec::new();
    bk_buf.extend_from_slice(BACKUP_MAGIC);
    bk_buf.extend_from_slice(&1u32.to_le_bytes());
    bk_buf.extend_from_slice(&(backup_records.len() as u32).to_le_bytes());
    for (kind, fname, off, data) in &backup_records {
        let fb = fname.as_bytes();
        bk_buf.push(*kind);
        bk_buf.extend_from_slice(&(fb.len() as u32).to_le_bytes());
        bk_buf.extend_from_slice(fb);
        bk_buf.extend_from_slice(&off.to_le_bytes());
        bk_buf.extend_from_slice(&(data.len() as u64).to_le_bytes());
        bk_buf.extend_from_slice(data);
    }
    fs::write(&backup_path, bk_buf).map_err(|e| e.to_string())?;

    let total_model_params: f64 = base_gguf
        .tensors
        .values()
        .map(|t| t.dims.iter().product::<usize>() as f64)
        .sum::<f64>()
        .max(1.0);
    let dof_pct = 100.0 * (subspace_dof as f64) / total_model_params;
    let mean_frob = if rel_deltas.is_empty() {
        0.0
    } else {
        rel_deltas.iter().sum::<f64>() / (rel_deltas.len() as f64)
    };
    let elapsed_s = t0.elapsed().as_secs_f64();

    let keys_j: Vec<String> = edited_keys.iter().map(|k| format!("\"{}\"", k)).collect();
    let layers_j: Vec<String> = layers.iter().map(|l| l.to_string()).collect();
    let manifest = format!(
        "{{\n  \"status\": \"patched\",\n  \"out_dir\": \"{}\",\n  \"gguf_path\": \"{}\",\n  \"layers\": [{}],\n  \"rank_k\": {},\n  \"alpha\": {:.4},\n  \"rescale_row_norms\": {},\n  \"edited_tensors\": [{}],\n  \"subspace_dof_params\": {},\n  \"subspace_dof_pct\": {:.6},\n  \"mean_rel_frobenius_delta\": {:.6},\n  \"elapsed_s\": {:.3}\n}}\n",
        out_ckpt_dir.display(),
        patched_gguf_path.display(),
        layers_j.join(", "),
        rank_k,
        alpha,
        rescale_row_norms,
        keys_j.join(", "),
        subspace_dof,
        dof_pct,
        mean_frob,
        elapsed_s
    );
    fs::write(out_ckpt_dir.join("patch_manifest.json"), manifest).map_err(|e| e.to_string())?;

    Ok(PatchSummary {
        out_dir: out_ckpt_dir.display().to_string(),
        gguf_path: patched_gguf_path.display().to_string(),
        edited_tensors: edited_keys,
        layers: layers.to_vec(),
        rank_k,
        alpha,
        rescale_row_norms,
        subspace_dof_params: subspace_dof,
        subspace_dof_pct: dof_pct,
        mean_rel_frobenius_delta: mean_frob,
        elapsed_s,
    })
}

/// Instant `<50 ms` rollback: restores every modified `BF16` `safetensors` shard and `Q8_0` GGUF slice
/// inside `ckpt_dir` to its exact pre-patch byte state!
pub fn rollback_surgical_weight_patch(ckpt_dir: &Path) -> Result<usize, String> {
    let backup_path = ckpt_dir.join("rollback_backup.bin");
    let raw = fs::read(&backup_path)
        .map_err(|e| format!("No rollback_backup.bin in {}: {}", ckpt_dir.display(), e))?;
    if raw.len() < 12 || &raw[0..4] != BACKUP_MAGIC {
        return Err("Invalid QMBK header in rollback_backup.bin".into());
    }
    let n_recs = get_u32(&raw, 8) as usize;
    let mut pos = 12usize;
    for _ in 0..n_recs {
        let _kind = raw[pos];
        pos += 1;
        let flen = get_u32(&raw, pos) as usize;
        pos += 4;
        let fname = std::str::from_utf8(&raw[pos..pos + flen]).map_err(|e| e.to_string())?;
        pos += flen;
        let file_off = get_u64(&raw, pos);
        pos += 8;
        let dlen = get_u64(&raw, pos) as usize;
        pos += 8;
        let data = &raw[pos..pos + dlen];
        pos += dlen;

        let target_file = ckpt_dir.join(fname);
        if target_file.exists() {
            let mut f = OpenOptions::new()
                .write(true)
                .open(&target_file)
                .map_err(|e| e.to_string())?;
            f.seek(SeekFrom::Start(file_off)).map_err(|e| e.to_string())?;
            f.write_all(data).map_err(|e| e.to_string())?;
        }
    }
    let _ = fs::remove_file(&backup_path);
    Ok(n_recs)
}

/// Exact `KL(P || Q) = sum_i P(i) * (log P(i) - log Q(i))` in nats over `151,936`-vocab FP32 GPU logits.
pub fn kl_divergence_from_logits(logits_p: &[f32], logits_q: &[f32]) -> f64 {
    assert_eq!(logits_p.len(), logits_q.len());
    let max_p = logits_p.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let max_q = logits_q.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum_exp_p: f64 = logits_p.iter().map(|&x| ((x as f64) - max_p).exp()).sum();
    let sum_exp_q: f64 = logits_q.iter().map(|&x| ((x as f64) - max_q).exp()).sum();
    let lse_p = max_p + sum_exp_p.ln();
    let lse_q = max_q + sum_exp_q.ln();

    let mut kl = 0.0f64;
    for i in 0..logits_p.len() {
        let lp = (logits_p[i] as f64) - lse_p;
        let lq = (logits_q[i] as f64) - lse_q;
        let p = lp.exp();
        if p > 1e-30 {
            kl += p * (lp - lq);
        }
    }
    kl.max(0.0)
}
