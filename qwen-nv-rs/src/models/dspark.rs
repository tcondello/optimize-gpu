//! Co-Resident `DSpark` (`Inkling2MTP`) Block-8 Multi-Token Prediction Drafter for `Inkling-2-Mini`.
//! Split-`Q8_0` 3-Layer Context Transformer + `BF16` Rank-256 Markov Head + GPU-Folded LM Head.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::gguf::f32_to_f16_bits;
use std::sync::atomic::{fence, Ordering};
use crate::hw::{align_up, get_u16, put_u64, CubinModule, HostMappedBuffer, NvGpu, VramBuffer};
use crate::safetensors::SafeTensorsFile;

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

const _: () = assert!(std::mem::size_of::<InkRunCtrl>() == 16576);
const _: () = assert!(std::mem::size_of::<InkDSparkLayerDesc>() == 128);
const _: () = assert!(std::mem::size_of::<InkDSparkDesc>() == 624);

pub(super) struct InkDSparkState {
    _weights_buf: VramBuffer,
    _draft_kv_buf: VramBuffer,
    _ws_buf: VramBuffer,
    _desc_buf: VramBuffer,
    desc_va: u64,
    pub total_vram_mb: f64,
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

pub(super) fn discover_mtp_safetensors(gguf_path: &Path) -> Option<PathBuf> {
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
    pub(super) fn load(
        gpu: &mut NvGpu,
        module: &CubinModule,
        mtp_path: &Path,
        vocab_unpadded: usize,
        max_ctx: usize,
        output_w_va: u64,
        output_w_type: i32,
    ) -> Result<Self, String> {
        let t0 = Instant::now();
        let st = SafeTensorsFile::open(mtp_path)?;
        let get_raw = |name: &str| -> Result<&[u8], String> { st.tensor_slice(name) };

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
        let mw1_crop_bytes = vocab_unpadded * 256 * 2;
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
        let fold_w_bytes = vocab_unpadded * 256 + vocab_unpadded * 8 * 2;
        let o_fold_w = align_up(cur_w_off, 256);
        cur_w_off = o_fold_w + (fold_w_bytes as u64);
        let fold_b_bytes = vocab_unpadded * 4;
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
        let per_kv_bytes = (3 * max_ctx * 1024 * 2) as u64;
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
        let d_base_logits = alloc_dws(7 * vocab_unpadded * 2);
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
        fold_args[28..32].copy_from_slice(&(vocab_unpadded as i32).to_le_bytes());
        put_u64(&mut fold_args, 32, dwb);
        put_u64(&mut fold_args, 40, dwb + hb_off);

        let k_fold = module.kernel("ink_dspark_build_fold_kernel")?;
        gpu.launch_kernel_sync(k_fold, (gpu.active_sms, 1, 1), (256, 1, 1), &fold_args, false, Duration::from_secs(15))?;

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
            max_ctx: max_ctx as i32,
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
            _weights_buf: weights_buf,
            _draft_kv_buf: draft_kv_buf,
            _ws_buf: ws_buf,
            _desc_buf: desc_buf,
            desc_va,
            total_vram_mb,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn inject_kv(
        &self,
        gpu: &mut NvGpu,
        module: &CubinModule,
        model_desc_va: u64,
        ctrl_host: &HostMappedBuffer,
        ws_bar_state_buf: &VramBuffer,
        expected_seq: u64,
        inject_pos: usize,
        inject_cnt: usize,
    ) -> Result<(), String> {
        if inject_cnt == 0 {
            return Ok(());
        }
        assert!(inject_cnt <= 8);
        let ctrl_ptr = ctrl_host.as_ptr() as *mut InkRunCtrl;
        unsafe {
            (*ctrl_ptr).cancel_flag = 0;
            (*ctrl_ptr).error_code = 0;
            (*ctrl_ptr).inject_pos = inject_pos as i32;
            (*ctrl_ptr).inject_cnt = inject_cnt as i32;
            (*ctrl_ptr).expected_seq = expected_seq;
        }
        fence(Ordering::SeqCst);

        let mut args = [0u8; 24];
        put_u64(&mut args, 0, model_desc_va);
        put_u64(&mut args, 8, self.desc_va);
        put_u64(&mut args, 16, ctrl_host.va_addr);

        let k_inj = module.kernel("ink_dspark_inject_kv_kernel")?;
        let err = gpu.launch_kernel_watched(
            k_inj,
            (gpu.active_sms, 1, 1),
            (512, 1, 1),
            &args,
            true,
            unsafe { &(*ctrl_ptr).error_code as *const i32 },
            unsafe { &mut (*ctrl_ptr).cancel_flag as *mut i32 },
            Duration::from_secs(10),
            "ink_dspark_inject_kv_kernel",
        )?;
        if err != 0 {
            let _ = gpu.zero_vram(ws_bar_state_buf);
            return Err(format!("ink_dspark_inject_kv_kernel barrier watchdog tripped at {}", err));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn propose(
        &self,
        gpu: &mut NvGpu,
        module: &CubinModule,
        model_desc_va: u64,
        ctrl_host: &HostMappedBuffer,
        ws_bar_state_buf: &VramBuffer,
        expected_seq: u64,
        anchor_tok: i32,
        anchor_pos: usize,
        bd_len: usize,
        req_gamma: usize,
        inject_pos: usize,
        inject_cnt: usize,
    ) -> Result<[i32; 8], String> {
        assert!((2..=8).contains(&bd_len));
        assert!(inject_cnt <= 8);
        let ctrl_ptr = ctrl_host.as_ptr() as *mut InkRunCtrl;
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
        put_u64(&mut args, 0, model_desc_va);
        put_u64(&mut args, 8, self.desc_va);
        put_u64(&mut args, 16, ctrl_host.va_addr);

        let k_prop = module.kernel("ink_dspark_propose_megakernel")?;
        let err = gpu.launch_kernel_watched(
            k_prop,
            (gpu.active_sms, 1, 1),
            (512, 1, 1),
            &args,
            true,
            unsafe { &(*ctrl_ptr).error_code as *const i32 },
            unsafe { &mut (*ctrl_ptr).cancel_flag as *mut i32 },
            Duration::from_secs(15),
            "ink_dspark_propose_megakernel",
        )?;
        if err != 0 {
            let _ = gpu.zero_vram(ws_bar_state_buf);
            return Err(format!("ink_dspark_propose_megakernel barrier watchdog tripped at {}", err));
        }
        let out = unsafe { std::ptr::read_volatile(&(*ctrl_ptr).batch_out_toks) };
        Ok(out)
    }
}
