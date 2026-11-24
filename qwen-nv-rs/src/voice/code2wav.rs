//! 100%-on-RTX-3080 `12Hz Code2Wav` Streaming Neural Audio Vocoder (`tok_dec.*`).
//! Split-RVQ + Causal Pre-Conv + 8-Layer Sliding-Window Transformer + 2x ConvNeXt Upsampler + 4-Block SnakeBeta DAC.

#![allow(dead_code)]

use std::sync::atomic::{fence, Ordering};
use std::time::Duration;

use super::audio::TTS_HOP_SAMPLES;
use crate::gguf::{decode_tensor_bf16_req, decode_tensor_f32_req, precompute_freqs_cis_f32, GgufFile};
use crate::hw::{put_u32, put_u64, CubinModule, HostMappedBuffer, NvGpu, VramBuffer};

pub const TTS_MAX_CODEC_CHUNK_FRAMES: usize = 32; // up to 32 frames (2.56s = 61,440 samples) per vocoder pass
pub(super) const C2W_TFM_MAX_CTX: usize = 2048;

pub(super) struct ByteSlab {
    bytes: Vec<u8>,
}

impl ByteSlab {
    pub(super) fn with_capacity(cap: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(cap),
        }
    }
    fn align256(&mut self) {
        let pad = (256 - (self.bytes.len() % 256)) % 256;
        self.bytes.resize(self.bytes.len() + pad, 0);
    }
    pub(super) fn push_bytes(&mut self, b: &[u8]) -> u64 {
        self.align256();
        let off = self.bytes.len() as u64;
        self.bytes.extend_from_slice(b);
        off
    }
    pub(super) fn push_f32(&mut self, v: &[f32]) -> u64 {
        self.align256();
        let off = self.bytes.len() as u64;
        for &x in v {
            self.bytes.extend_from_slice(&x.to_le_bytes());
        }
        off
    }
    pub(super) fn push_u16(&mut self, v: &[u16]) -> u64 {
        self.align256();
        let off = self.bytes.len() as u64;
        for &x in v {
            self.bytes.extend_from_slice(&x.to_le_bytes());
        }
        off
    }
    #[inline]
    pub(super) fn as_slice(&self) -> &[u8] {
        &self.bytes
    }
}

#[derive(Clone, Copy)]
struct C2wTfmLayer {
    attn_norm_w: u64,
    w_qkv: u64,
    w_o: u64,
    attn_scale: u64,
    ffn_norm_w: u64,
    w_gate_up: u64,
    w_down: u64,
    ffn_scale: u64,
    kv_cache: u64,
}

#[derive(Clone, Copy)]
struct C2wUpsampleBlk {
    tc_w: u64, // [2 * 1024, 1024] BF16
    tc_b: u64, // [1024] F32
    dw_w: u64, // [1024, 7] F32
    dw_b: u64, // [1024] F32
    dw_state: u64, // [6, 1024] F32
    norm_w: u64,
    norm_b: u64,
    pw1_w: u64, // [4096, 1024] BF16
    pw1_b: u64,
    pw2_w: u64, // [1024, 4096] BF16
    pw2_b: u64,
    gamma: u64,
}

#[derive(Clone, Copy)]
struct C2wResUnit {
    act1_a: u64,
    act1_inv_b: u64,
    c1_w: u64,     // [7, ch, ch] BF16
    c1_b: u64,
    c1_state: u64, // [6 * dilation, ch] F32
    act2_a: u64,
    act2_inv_b: u64,
    c2_w: u64,     // [1, ch, ch] BF16
    c2_b: u64,
    dilation: i32,
}

#[derive(Clone, Copy)]
struct C2wDacBlk {
    snake_a: u64,
    snake_inv_b: u64,
    tc_w: u64,     // [out_ch * kernel, in_ch] BF16
    tc_b: u64,
    tc_carry: u64, // [stride, out_ch] F32
    res: [C2wResUnit; 3],
    in_ch: usize,
    out_ch: usize,
    stride: usize,
    kernel: usize,
}

pub(super) struct Code2WavEngine {
    _w_vram: VramBuffer,
    vq_first_cb: u64,
    vq_rest_cb: u64,
    proj_first_w: u64,
    proj_rest_w: u64,

    pre_conv_w: u64,
    pre_conv_b: u64,
    pre_conv_state: u64, // [2, 512]

    tfm_in_w: u64,
    tfm_in_b: u64,
    tfm_layers: [C2wTfmLayer; 8],
    tfm_norm_w: u64,
    tfm_out_w: u64,
    tfm_out_b: u64,
    tfm_freqs_cis: u64,
    tfm_sliding_window: i32,
    tfm_eps: f32,

    upsample: [C2wUpsampleBlk; 2],

    dac_pre_w: u64,
    dac_pre_b: u64,
    dac_pre_state: u64, // [6, 1024]
    dac_blks: [C2wDacBlk; 4],
    dac_post_snake_a: u64,
    dac_post_snake_inv_b: u64,
    dac_post_w: u64,
    dac_post_b: u64,
    dac_post_state: u64, // [6, 96]

    state_slab: VramBuffer,
    kv_slab: VramBuffer,
    tmp_state: VramBuffer,
    buf0: VramBuffer, // 52 MB activation ping-pong A / col
    buf1: VramBuffer, // 32 MB activation ping-pong B
    buf2: VramBuffer, // 32 MB residual / skip / q_roped
    stream_pos: usize,
}

impl Code2WavEngine {
    pub(super) fn load(
        gpu: &mut NvGpu,
        gf: &GgufFile,
        total_vram_bytes: &mut u64,
    ) -> Result<Self, String> {
        let mut w_slab = ByteSlab::with_capacity(240 << 20);

        // Permute Conv1d `dims=[K, IC, OC]` (`raw[(oc * IC + ic) * K + k]`) -> `[K, OC, IC]` (`dst[(k * OC + oc) * IC + ic]`)
        let push_conv1d_bf16 = |slab: &mut ByteSlab, name: &str| -> Result<u64, String> {
            let (raw, dims) = decode_tensor_bf16_req(gf, name)?;
            let k_sz = dims[0];
            let ic_sz = dims[1];
            let oc_sz = if dims.len() > 2 { dims[2] } else { 1 };
            let mut perm = vec![0u16; k_sz * oc_sz * ic_sz];
            for oc in 0..oc_sz {
                for ic in 0..ic_sz {
                    for k in 0..k_sz {
                        perm[(k * oc_sz + oc) * ic_sz + ic] = raw[(oc * ic_sz + ic) * k_sz + k];
                    }
                }
            }
            Ok(slab.push_u16(&perm))
        };

        // Permute ConvTranspose1d `dims=[K, OC, IC]` (`raw[(ic * OC + oc) * K + k]`) -> `[OC * K, IC]` (`dst[(oc * K + k) * IC + ic]`)
        let push_transconv_bf16 = |slab: &mut ByteSlab, name: &str| -> Result<u64, String> {
            let (raw, dims) = decode_tensor_bf16_req(gf, name)?;
            let k_sz = dims[0];
            let oc_sz = dims[1];
            let ic_sz = dims[2];
            let mut perm = vec![0u16; oc_sz * k_sz * ic_sz];
            for ic in 0..ic_sz {
                for oc in 0..oc_sz {
                    for k in 0..k_sz {
                        perm[(oc * k_sz + k) * ic_sz + ic] = raw[(ic * oc_sz + oc) * k_sz + k];
                    }
                }
            }
            Ok(slab.push_u16(&perm))
        };

        let push_linear_bf16 = |slab: &mut ByteSlab, name: &str| -> Result<u64, String> {
            let (raw, _) = decode_tensor_bf16_req(gf, name)?;
            Ok(slab.push_u16(&raw))
        };

        let push_f32_req = |slab: &mut ByteSlab, name: &str| -> Result<u64, String> {
            let (v, _) = decode_tensor_f32_req(gf, name)?;
            Ok(slab.push_f32(&v))
        };

        let push_snake = |slab: &mut ByteSlab, alpha_name: &str, beta_name: &str| -> Result<(u64, u64), String> {
            let (alpha, _) = decode_tensor_f32_req(gf, alpha_name)?;
            let (beta, _) = decode_tensor_f32_req(gf, beta_name)?;
            let a: Vec<f32> = alpha.iter().map(|&x| x.exp()).collect();
            let inv_b: Vec<f32> = beta.iter().map(|&x| 1.0 / (x.exp() + 1e-9)).collect();
            Ok((slab.push_f32(&a), slab.push_f32(&inv_b)))
        };

        // 1. Split RVQ
        let vq_first_off = push_f32_req(&mut w_slab, "tok_dec.vq_first.0.codebook")?;
        let mut rest_all = Vec::with_capacity(15 * 2048 * 256);
        for g in 0..15 {
            let (cb, _) = decode_tensor_f32_req(gf, &format!("tok_dec.vq_rest.{}.codebook", g))?;
            rest_all.extend_from_slice(&cb);
        }
        let vq_rest_off = w_slab.push_f32(&rest_all);
        let proj_first_off = push_f32_req(&mut w_slab, "tok_dec.vq_first.output_proj.weight")?;
        let proj_rest_off = push_f32_req(&mut w_slab, "tok_dec.vq_rest.output_proj.weight")?;

        // 2. pre_conv
        let pre_conv_w_off = push_conv1d_bf16(&mut w_slab, "tok_dec.pre_conv.weight")?;
        let pre_conv_b_off = push_f32_req(&mut w_slab, "tok_dec.pre_conv.bias")?;

        // 3. 8-Layer Sliding-Window Transformer
        let tfm_sliding_window = gf.get_usize_or("qwen3-tts-tokenizer.decoder.sliding_window", 72) as i32;
        let tfm_rope_theta = gf.get_f32_or("qwen3-tts-tokenizer.decoder.rope_theta", 10_000.0);
        let tfm_eps = gf.get_f32_or("qwen3-tts-tokenizer.decoder.rms_norm_eps", 1e-5);

        let tfm_freqs_off = w_slab.push_f32(&precompute_freqs_cis_f32(64, C2W_TFM_MAX_CTX, tfm_rope_theta));
        let tfm_in_w_off = push_linear_bf16(&mut w_slab, "tok_dec.pre_tfm.input_proj.weight")?;
        let tfm_in_b_off = push_f32_req(&mut w_slab, "tok_dec.pre_tfm.input_proj.bias")?;
        let tfm_norm_w_off = push_f32_req(&mut w_slab, "tok_dec.pre_tfm.norm.weight")?;
        let tfm_out_w_off = push_linear_bf16(&mut w_slab, "tok_dec.pre_tfm.output_proj.weight")?;
        let tfm_out_b_off = push_f32_req(&mut w_slab, "tok_dec.pre_tfm.output_proj.bias")?;

        let mut tfm_offs = [(0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64); 8];
        for l in 0..8 {
            let p = format!("tok_dec.pre_tfm.blk.{}", l);
            let an = push_f32_req(&mut w_slab, &format!("{}.attn_norm.weight", p))?;
            let (wq, _) = decode_tensor_bf16_req(gf, &format!("{}.attn_q.weight", p))?;
            let (wk, _) = decode_tensor_bf16_req(gf, &format!("{}.attn_k.weight", p))?;
            let (wv, _) = decode_tensor_bf16_req(gf, &format!("{}.attn_v.weight", p))?;
            let mut qkv = Vec::with_capacity(wq.len() + wk.len() + wv.len());
            qkv.extend_from_slice(&wq);
            qkv.extend_from_slice(&wk);
            qkv.extend_from_slice(&wv);
            let qkv_off = w_slab.push_u16(&qkv);
            let wo = push_linear_bf16(&mut w_slab, &format!("{}.attn_output.weight", p))?;
            let asc = push_f32_req(&mut w_slab, &format!("{}.attn_scale", p))?;
            let fnw = push_f32_req(&mut w_slab, &format!("{}.ffn_norm.weight", p))?;
            let (wg, _) = decode_tensor_bf16_req(gf, &format!("{}.ffn_gate.weight", p))?;
            let (wu, _) = decode_tensor_bf16_req(gf, &format!("{}.ffn_up.weight", p))?;
            let mut gu = Vec::with_capacity(wg.len() + wu.len());
            gu.extend_from_slice(&wg);
            gu.extend_from_slice(&wu);
            let gu_off = w_slab.push_u16(&gu);
            let wd = push_linear_bf16(&mut w_slab, &format!("{}.ffn_down.weight", p))?;
            let fsc = push_f32_req(&mut w_slab, &format!("{}.ffn_scale", p))?;
            tfm_offs[l] = (an, qkv_off, wo, asc, fnw, gu_off, wd, fsc);
        }

        // 4. 2x Upsample + ConvNeXt
        let mut up_offs = [(0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64); 2];
        for i in 0..2 {
            let p = format!("tok_dec.upsample.{}", i);
            up_offs[i] = (
                push_transconv_bf16(&mut w_slab, &format!("{}.conv.weight", p))?,
                push_f32_req(&mut w_slab, &format!("{}.conv.bias", p))?,
                push_f32_req(&mut w_slab, &format!("{}.dwconv.weight", p))?,
                push_f32_req(&mut w_slab, &format!("{}.dwconv.bias", p))?,
                push_f32_req(&mut w_slab, &format!("{}.norm.weight", p))?,
                push_f32_req(&mut w_slab, &format!("{}.norm.bias", p))?,
                push_linear_bf16(&mut w_slab, &format!("{}.pwconv1.weight", p))?,
                push_f32_req(&mut w_slab, &format!("{}.pwconv1.bias", p))?,
                push_linear_bf16(&mut w_slab, &format!("{}.pwconv2.weight", p))?,
                push_f32_req(&mut w_slab, &format!("{}.pwconv2.bias", p))?,
                push_f32_req(&mut w_slab, &format!("{}.gamma", p))?,
            );
        }

        // 5. DAC v2 SnakeBeta Decoder
        let dac_pre_w_off = push_conv1d_bf16(&mut w_slab, "tok_dec.dec.0.conv.weight")?;
        let dac_pre_b_off = push_f32_req(&mut w_slab, "tok_dec.dec.0.conv.bias")?;

        let strides = [8usize, 5, 4, 3];
        let chs = [1536usize, 768, 384, 192, 96];
        let dilations = [1i32, 3, 9];

        #[derive(Clone, Copy, Default)]
        struct RuOff {
            a1: u64, ib1: u64, c1w: u64, c1b: u64,
            a2: u64, ib2: u64, c2w: u64, c2b: u64,
        }
        #[derive(Clone, Copy, Default)]
        struct DacOff {
            sa: u64, sib: u64, tcw: u64, tcb: u64, ru: [RuOff; 3],
        }
        let mut dac_offs = [DacOff::default(); 4];
        for i in 0..4 {
            let p = format!("tok_dec.dec.{}", i + 1);
            let (sa, sib) = push_snake(&mut w_slab, &format!("{}.snake.alpha", p), &format!("{}.snake.beta", p))?;
            let tcw = push_transconv_bf16(&mut w_slab, &format!("{}.conv_t.weight", p))?;
            let tcb = push_f32_req(&mut w_slab, &format!("{}.conv_t.bias", p))?;
            let mut ru = [RuOff::default(); 3];
            for r in 0..3 {
                let rp = format!("{}.res.{}", p, r);
                let (a1, ib1) = push_snake(&mut w_slab, &format!("{}.act1.alpha", rp), &format!("{}.act1.beta", rp))?;
                let c1w = push_conv1d_bf16(&mut w_slab, &format!("{}.conv1.weight", rp))?;
                let c1b = push_f32_req(&mut w_slab, &format!("{}.conv1.bias", rp))?;
                let (a2, ib2) = push_snake(&mut w_slab, &format!("{}.act2.alpha", rp), &format!("{}.act2.beta", rp))?;
                let c2w = push_conv1d_bf16(&mut w_slab, &format!("{}.conv2.weight", rp))?;
                let c2b = push_f32_req(&mut w_slab, &format!("{}.conv2.bias", rp))?;
                ru[r] = RuOff { a1, ib1, c1w, c1b, a2, ib2, c2w, c2b };
            }
            dac_offs[i] = DacOff { sa, sib, tcw, tcb, ru };
        }
        let (post_sa, post_sib) = push_snake(&mut w_slab, "tok_dec.dec.5.snake.alpha", "tok_dec.dec.5.snake.beta")?;
        let post_w_off = push_conv1d_bf16(&mut w_slab, "tok_dec.dec.6.conv.weight")?;
        let post_b_off = push_f32_req(&mut w_slab, "tok_dec.dec.6.conv.bias")?;

        let c2w_w_vram = gpu.alloc_vram(w_slab.as_slice().len() as u64)?;
        *total_vram_bytes += c2w_w_vram.size;
        gpu.upload_to_vram(&c2w_w_vram, 0, &w_slab.as_slice())?;
        let base_va = c2w_w_vram.va_addr;
        drop(w_slab);

        // Allocate streaming conv/carry state slab (`~1.5 MB`) + Transformer KV slab (`67 MB`) + 3 ping-pong buffers
        let mut st_off = 0u64;
        let mut alloc_st = |floats: usize| -> u64 {
            let aligned = (st_off + 255) & !255;
            st_off = aligned + (floats * 4) as u64;
            aligned
        };
        let pre_conv_st_rel = alloc_st(2 * 512);
        let up0_st_rel = alloc_st(6 * 1024);
        let up1_st_rel = alloc_st(6 * 1024);
        let dac_pre_st_rel = alloc_st(6 * 1024);
        let mut dac_carry_rel = [0u64; 4];
        let mut dac_ru_rel = [[0u64; 3]; 4];
        for i in 0..4 {
            dac_carry_rel[i] = alloc_st(strides[i] * chs[i + 1]);
            for r in 0..3 {
                dac_ru_rel[i][r] = alloc_st((6 * dilations[r] as usize) * chs[i + 1]);
            }
        }
        let dac_post_st_rel = alloc_st(6 * 96);

        let state_slab = gpu.alloc_vram(st_off.max(65536))?;
        *total_vram_bytes += state_slab.size;
        gpu.zero_vram(&state_slab)?;
        let st_va = state_slab.va_addr;

        let tfm_kv_layer_bytes = ((2 * 16 * C2W_TFM_MAX_CTX * 64 * 4) as u64 + 255) & !255;
        let kv_slab = gpu.alloc_vram(tfm_kv_layer_bytes * 8)?;
        *total_vram_bytes += kv_slab.size;
        let tfm_kv_va = kv_slab.va_addr;

        let tmp_state = gpu.alloc_vram(512 * 1024)?; // 512 KB staging for race-free state shift
        *total_vram_bytes += tmp_state.size;

        let buf0 = gpu.alloc_vram(52 << 20)?; // 52 MB (fits `col [20480, 576]` = 47.2 MB!)
        let buf1 = gpu.alloc_vram(32 << 20)?; // 32 MB
        let buf2 = gpu.alloc_vram(32 << 20)?; // 32 MB
        *total_vram_bytes += buf0.size + buf1.size + buf2.size;

        let mut tfm_layers = [C2wTfmLayer {
            attn_norm_w: 0, w_qkv: 0, w_o: 0, attn_scale: 0,
            ffn_norm_w: 0, w_gate_up: 0, w_down: 0, ffn_scale: 0, kv_cache: 0,
        }; 8];
        for l in 0..8 {
            let t = tfm_offs[l];
            tfm_layers[l] = C2wTfmLayer {
                attn_norm_w: base_va + t.0,
                w_qkv: base_va + t.1,
                w_o: base_va + t.2,
                attn_scale: base_va + t.3,
                ffn_norm_w: base_va + t.4,
                w_gate_up: base_va + t.5,
                w_down: base_va + t.6,
                ffn_scale: base_va + t.7,
                kv_cache: tfm_kv_va + (l as u64) * tfm_kv_layer_bytes,
            };
        }

        let mk_up = |o: (u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64), st_rel: u64| C2wUpsampleBlk {
            tc_w: base_va + o.0,
            tc_b: base_va + o.1,
            dw_w: base_va + o.2,
            dw_b: base_va + o.3,
            dw_state: st_va + st_rel,
            norm_w: base_va + o.4,
            norm_b: base_va + o.5,
            pw1_w: base_va + o.6,
            pw1_b: base_va + o.7,
            pw2_w: base_va + o.8,
            pw2_b: base_va + o.9,
            gamma: base_va + o.10,
        };
        let upsample = [mk_up(up_offs[0], up0_st_rel), mk_up(up_offs[1], up1_st_rel)];

        let mut dac_blks = [C2wDacBlk {
            snake_a: 0, snake_inv_b: 0, tc_w: 0, tc_b: 0, tc_carry: 0,
            res: [C2wResUnit {
                act1_a: 0, act1_inv_b: 0, c1_w: 0, c1_b: 0, c1_state: 0,
                act2_a: 0, act2_inv_b: 0, c2_w: 0, c2_b: 0, dilation: 1,
            }; 3],
            in_ch: 0, out_ch: 0, stride: 0, kernel: 0,
        }; 4];
        for i in 0..4 {
            let d = dac_offs[i];
            let mut res = [dac_blks[0].res[0]; 3];
            for r in 0..3 {
                let ru = d.ru[r];
                res[r] = C2wResUnit {
                    act1_a: base_va + ru.a1,
                    act1_inv_b: base_va + ru.ib1,
                    c1_w: base_va + ru.c1w,
                    c1_b: base_va + ru.c1b,
                    c1_state: st_va + dac_ru_rel[i][r],
                    act2_a: base_va + ru.a2,
                    act2_inv_b: base_va + ru.ib2,
                    c2_w: base_va + ru.c2w,
                    c2_b: base_va + ru.c2b,
                    dilation: dilations[r],
                };
            }
            dac_blks[i] = C2wDacBlk {
                snake_a: base_va + d.sa,
                snake_inv_b: base_va + d.sib,
                tc_w: base_va + d.tcw,
                tc_b: base_va + d.tcb,
                tc_carry: st_va + dac_carry_rel[i],
                res,
                in_ch: chs[i],
                out_ch: chs[i + 1],
                stride: strides[i],
                kernel: 2 * strides[i],
            };
        }

        Ok(Self {
            _w_vram: c2w_w_vram,
            vq_first_cb: base_va + vq_first_off,
            vq_rest_cb: base_va + vq_rest_off,
            proj_first_w: base_va + proj_first_off,
            proj_rest_w: base_va + proj_rest_off,
            pre_conv_w: base_va + pre_conv_w_off,
            pre_conv_b: base_va + pre_conv_b_off,
            pre_conv_state: st_va + pre_conv_st_rel,
            tfm_in_w: base_va + tfm_in_w_off,
            tfm_in_b: base_va + tfm_in_b_off,
            tfm_layers,
            tfm_norm_w: base_va + tfm_norm_w_off,
            tfm_out_w: base_va + tfm_out_w_off,
            tfm_out_b: base_va + tfm_out_b_off,
            tfm_freqs_cis: base_va + tfm_freqs_off,
            tfm_sliding_window,
            tfm_eps,
            upsample,
            dac_pre_w: base_va + dac_pre_w_off,
            dac_pre_b: base_va + dac_pre_b_off,
            dac_pre_state: st_va + dac_pre_st_rel,
            dac_blks,
            dac_post_snake_a: base_va + post_sa,
            dac_post_snake_inv_b: base_va + post_sib,
            dac_post_w: base_va + post_w_off,
            dac_post_b: base_va + post_b_off,
            dac_post_state: st_va + dac_post_st_rel,
            state_slab,
            kv_slab,
            tmp_state,
            buf0,
            buf1,
            buf2,
            stream_pos: 0,
        })
    }

    pub(super) fn reset_stream(&mut self, gpu: &mut NvGpu) -> Result<(), String> {
        gpu.zero_vram(&self.state_slab)?;
        self.stream_pos = 0;
        Ok(())
    }

    fn execute_qmd_chain(gpu: &mut NvGpu, chain: &[(u64, usize)], timeout_secs: u64) -> Result<(), String> {
        gpu.launch_qmd_chain_sync(chain, true, Duration::from_secs(timeout_secs))
    }

    /// Decode `[n_frames, 16]` RVQ codes (`1 <= n_frames <= 32` per sub-batch) 100% on RTX 3080!
    pub(super) fn decode_stream_chunk(
        &mut self,
        gpu: &mut NvGpu,
        module: &CubinModule,
        host_io_buf: &mut HostMappedBuffer,
        codes_flat: &[i32],
        n_frames: usize,
    ) -> Result<Vec<f32>, String> {
        if n_frames == 0 {
            return Ok(Vec::new());
        }
        assert_eq!(codes_flat.len(), n_frames * 16);

        let mut all_samples = Vec::with_capacity(n_frames * TTS_HOP_SAMPLES);
        let mut f_off = 0usize;

        while f_off < n_frames {
            let t_in = (n_frames - f_off).min(TTS_MAX_CODEC_CHUNK_FRAMES);
            if self.stream_pos + t_in > C2W_TFM_MAX_CTX {
                return Err("Code2Wav streaming context exceeded 2048 frames (163s)".into());
            }

            // Stage `[t_in * 16]` codes into `host_io_buf[0..]`
            {
                let dst = host_io_buf.as_mut_slice();
                let slice = &codes_flat[f_off * 16..(f_off + t_in) * 16];
                for (i, &c) in slice.iter().enumerate() {
                    put_u32(dst, i * 4, c as u32);
                }
            }
            fence(Ordering::SeqCst);

            let codes_va = host_io_buf.va_addr;
            let audio_out_va = host_io_buf.va_addr + 16384;

            let k_rvq = module.kernel("c2w_rvq_decode_kernel")?;
            let k_conv = module.kernel("c2w_causal_conv1d_bf16_kernel")?;
            let k_st_stage = module.kernel("c2w_conv_state_stage_kernel")?;
            let k_copy = module.kernel("tts_copy_f32")?;
            let k_lin = module.kernel("c2w_linear_bf16_kernel")?;
            let k_rms = module.kernel("c2w_rmsnorm_f32_kernel")?;
            let k_rope = module.kernel("c2w_tfm_rope_kv_write_kernel")?;
            let k_attn = module.kernel("c2w_tfm_sliding_attn_kernel")?;
            let k_swiglu = module.kernel("c2w_swiglu_f32_kernel")?;
            let k_col2im = module.kernel("c2w_transconv_col2im_causal_kernel")?;
            let k_carry = module.kernel("c2w_transconv_carry_update_kernel")?;
            let k_dw = module.kernel("c2w_dwconv1d_k7_causal_kernel")?;
            let k_ln = module.kernel("c2w_layernorm_f32_kernel")?;
            let k_gelu = module.kernel("c2w_gelu_inplace_f32_kernel")?;
            let k_snake = module.kernel("c2w_snake_beta_kernel")?;
            let k_clamp = module.kernel("c2w_clamp_audio_kernel")?;

            let b0 = self.buf0.va_addr;
            let b1 = self.buf1.va_addr;
            let b2 = self.buf2.va_addr;
            let tmp_st = self.tmp_state.va_addr;

            // Total QMDs in one Code2Wav chunk is ~155 -> fits easily in our 4MB kernargs ring (`1` single doorbell!)
            let mut chain = Vec::<(u64, usize)>::with_capacity(180);

            let push_lin = |gpu: &mut NvGpu,
                            chain: &mut Vec<(u64, usize)>,
                            out_va: u64,
                            w_va: u64,
                            x_va: u64,
                            bias_va: u64,
                            ls_va: u64,
                            res_va: u64,
                            t_rows: usize,
                            n_out: usize,
                            k_in: usize| {
                let mut a = [0u8; 60];
                put_u64(&mut a, 0, out_va);
                put_u64(&mut a, 8, w_va);
                put_u64(&mut a, 16, x_va);
                put_u64(&mut a, 24, bias_va);
                put_u64(&mut a, 32, ls_va);
                put_u64(&mut a, 40, res_va);
                put_u32(&mut a, 48, t_rows as u32);
                put_u32(&mut a, 52, n_out as u32);
                put_u32(&mut a, 56, k_in as u32);
                chain.push(gpu.prepare_qmd(
                    k_lin,
                    (((n_out as u32) + 3) / 4, ((t_rows as u32) + 7) / 8, 1),
                    (128, 1, 1),
                    &a,
                    None,
                ));
            };

            let push_causal_conv = |gpu: &mut NvGpu,
                                    chain: &mut Vec<(u64, usize)>,
                                    out_va: u64,
                                    w_va: u64,
                                    x_va: u64,
                                    state_va: u64,
                                    bias_va: u64,
                                    skip_va: u64,
                                    t_rows: usize,
                                    ic: usize,
                                    oc: usize,
                                    k_sz: usize,
                                    dil: i32| {
                let mut a = [0u8; 68];
                put_u64(&mut a, 0, out_va);
                put_u64(&mut a, 8, w_va);
                put_u64(&mut a, 16, x_va);
                put_u64(&mut a, 24, state_va);
                put_u64(&mut a, 32, bias_va);
                put_u64(&mut a, 40, skip_va);
                put_u32(&mut a, 48, t_rows as u32);
                put_u32(&mut a, 52, ic as u32);
                put_u32(&mut a, 56, oc as u32);
                put_u32(&mut a, 60, k_sz as u32);
                put_u32(&mut a, 64, dil as u32);
                chain.push(gpu.prepare_qmd(
                    k_conv,
                    (((oc as u32) + 3) / 4, ((t_rows as u32) + 7) / 8, 1),
                    (128, 1, 1),
                    &a,
                    None,
                ));

                let l_rows = (k_sz - 1) * (dil as usize);
                if l_rows > 0 && state_va != 0 {
                    let tot = (l_rows * ic) as u32;
                    let mut sa = [0u8; 36];
                    put_u64(&mut sa, 0, tmp_st);
                    put_u64(&mut sa, 8, state_va);
                    put_u64(&mut sa, 16, x_va);
                    put_u32(&mut sa, 24, t_rows as u32);
                    put_u32(&mut sa, 28, ic as u32);
                    put_u32(&mut sa, 32, l_rows as u32);
                    chain.push(gpu.prepare_qmd(k_st_stage, ((tot + 255) / 256, 1, 1), (256, 1, 1), &sa, None));

                    let mut ca = [0u8; 20];
                    put_u64(&mut ca, 0, state_va);
                    put_u64(&mut ca, 8, tmp_st);
                    put_u32(&mut ca, 16, tot);
                    chain.push(gpu.prepare_qmd(k_copy, ((tot + 255) / 256, 1, 1), (256, 1, 1), &ca, None));
                }
            };

            let push_snake = |gpu: &mut NvGpu,
                              chain: &mut Vec<(u64, usize)>,
                              x_va: u64,
                              a_va: u64,
                              ib_va: u64,
                              t_rows: usize,
                              ch: usize| {
                let tot = (t_rows * ch) as u32;
                let mut a = [0u8; 32];
                put_u64(&mut a, 0, x_va);
                put_u64(&mut a, 8, a_va);
                put_u64(&mut a, 16, ib_va);
                put_u32(&mut a, 24, t_rows as u32);
                put_u32(&mut a, 28, ch as u32);
                chain.push(gpu.prepare_qmd(k_snake, ((tot + 255) / 256, 1, 1), (256, 1, 1), &a, None));
            };

            // Stage 1: Split RVQ -> `b0 [t_in, 512]`
            {
                let mut a = [0u8; 52];
                put_u64(&mut a, 0, b0);
                put_u64(&mut a, 8, codes_va);
                put_u64(&mut a, 16, self.vq_first_cb);
                put_u64(&mut a, 24, self.vq_rest_cb);
                put_u64(&mut a, 32, self.proj_first_w);
                put_u64(&mut a, 40, self.proj_rest_w);
                put_u32(&mut a, 48, t_in as u32);
                chain.push(gpu.prepare_qmd(k_rvq, (t_in as u32, 2, 1), (256, 1, 1), &a, None));
            }

            // Stage 2: `pre_conv` (`b0 [t_in, 512]` -> `b1 [t_in, 1024]`, k=3, d=1)
            push_causal_conv(
                gpu,
                &mut chain,
                b1,
                self.pre_conv_w,
                b0,
                self.pre_conv_state,
                self.pre_conv_b,
                0,
                t_in,
                512,
                1024,
                3,
                1,
            );

            // Stage 3: 8-Layer Causal Sliding-Window Transformer (`b1 [t_in, 1024]` -> `b0 [t_in, 512]` -> `b1 [t_in, 1024]`)
            push_lin(gpu, &mut chain, b0, self.tfm_in_w, b1, self.tfm_in_b, 0, 0, t_in, 512, 1024);

            let tfm_eps_bits = self.tfm_eps.to_bits();
            let attn_scale_bits = (1.0f32 / 8.0f32).to_bits(); // 1 / sqrt(64)
            let start_pos = self.stream_pos as u32;

            for l in 0..8 {
                let tl = self.tfm_layers[l];
                // RMSNorm `b0 -> b1`
                let mut a_rn1 = [0u8; 36];
                put_u64(&mut a_rn1, 0, b1);
                put_u64(&mut a_rn1, 8, b0);
                put_u64(&mut a_rn1, 16, tl.attn_norm_w);
                put_u32(&mut a_rn1, 24, t_in as u32);
                put_u32(&mut a_rn1, 28, 512);
                put_u32(&mut a_rn1, 32, tfm_eps_bits);
                chain.push(gpu.prepare_qmd(k_rms, (t_in as u32, 1, 1), (256, 1, 1), &a_rn1, None));

                // QKV (`b1 [t_in, 512]` -> `b2 [t_in, 3072]`)
                push_lin(gpu, &mut chain, b2, tl.w_qkv, b1, 0, 0, 0, t_in, 3072, 512);

                // RoPE + KV write (`b2` -> `q_out=b1 [16, t_in, 64]` + `tl.kv_cache`)
                let mut a_rp = [0u8; 48];
                put_u64(&mut a_rp, 0, b1);
                put_u64(&mut a_rp, 8, tl.kv_cache);
                put_u64(&mut a_rp, 16, b2);
                put_u64(&mut a_rp, 24, self.tfm_freqs_cis);
                put_u32(&mut a_rp, 32, t_in as u32);
                put_u32(&mut a_rp, 36, start_pos);
                put_u32(&mut a_rp, 40, 16);
                put_u32(&mut a_rp, 44, C2W_TFM_MAX_CTX as u32);
                chain.push(gpu.prepare_qmd(k_rope, (48, t_in as u32, 1), (64, 1, 1), &a_rp, None));

                // Sliding-window attention (`q=b1` -> `attn_out=b2 [t_in, 1024]`)
                let mut a_at = [0u8; 48];
                put_u64(&mut a_at, 0, b2);
                put_u64(&mut a_at, 8, b1);
                put_u64(&mut a_at, 16, tl.kv_cache);
                put_u32(&mut a_at, 24, t_in as u32);
                put_u32(&mut a_at, 28, start_pos);
                put_u32(&mut a_at, 32, 16);
                put_u32(&mut a_at, 36, C2W_TFM_MAX_CTX as u32);
                put_u32(&mut a_at, 40, self.tfm_sliding_window as u32);
                put_u32(&mut a_at, 44, attn_scale_bits);
                chain.push(gpu.prepare_qmd(k_attn, (16, t_in as u32, 1), (128, 1, 1), &a_at, None));

                // O-proj + LayerScale + Residual (`b2 [t_in, 1024]`, res=`b0` -> `b1 [t_in, 512]`)
                push_lin(gpu, &mut chain, b1, tl.w_o, b2, 0, tl.attn_scale, b0, t_in, 512, 1024);

                // FFN RMSNorm (`b1 -> b2`)
                let mut a_rn2 = [0u8; 36];
                put_u64(&mut a_rn2, 0, b2);
                put_u64(&mut a_rn2, 8, b1);
                put_u64(&mut a_rn2, 16, tl.ffn_norm_w);
                put_u32(&mut a_rn2, 24, t_in as u32);
                put_u32(&mut a_rn2, 28, 512);
                put_u32(&mut a_rn2, 32, tfm_eps_bits);
                chain.push(gpu.prepare_qmd(k_rms, (t_in as u32, 1, 1), (256, 1, 1), &a_rn2, None));

                // Gate+Up (`b2 [t_in, 512]` -> `b0 [t_in, 2048]`) — note residual is safely in `b1`!
                push_lin(gpu, &mut chain, b0, tl.w_gate_up, b2, 0, 0, 0, t_in, 2048, 512);

                // SwiGLU (`b0 [t_in, 2048]` -> `b2 [t_in, 1024]`)
                let tot_sw = (t_in * 1024) as u32;
                let mut a_sw = [0u8; 24];
                put_u64(&mut a_sw, 0, b2);
                put_u64(&mut a_sw, 8, b0);
                put_u32(&mut a_sw, 16, t_in as u32);
                put_u32(&mut a_sw, 20, 1024);
                chain.push(gpu.prepare_qmd(k_swiglu, ((tot_sw + 255) / 256, 1, 1), (256, 1, 1), &a_sw, None));

                // Down + LayerScale + Residual (`b2`, res=`b1` -> `b0 [t_in, 512]`)
                push_lin(gpu, &mut chain, b0, tl.w_down, b2, 0, tl.ffn_scale, b1, t_in, 512, 1024);
            }

            // Final Transformer Norm (`b0 -> b1`) + `output_proj` (`b1 -> b2 [t_in, 1024]`)
            {
                let mut a_rnf = [0u8; 36];
                put_u64(&mut a_rnf, 0, b1);
                put_u64(&mut a_rnf, 8, b0);
                put_u64(&mut a_rnf, 16, self.tfm_norm_w);
                put_u32(&mut a_rnf, 24, t_in as u32);
                put_u32(&mut a_rnf, 28, 512);
                put_u32(&mut a_rnf, 32, tfm_eps_bits);
                chain.push(gpu.prepare_qmd(k_rms, (t_in as u32, 1, 1), (256, 1, 1), &a_rnf, None));
            }
            push_lin(gpu, &mut chain, b2, self.tfm_out_w, b1, self.tfm_out_b, 0, 0, t_in, 1024, 512);

            // Stage 4: 2x Upsample + ConvNeXt (`cur` starts in `b2`, `t_cur = t_in -> t_in*2 -> t_in*4`)
            let mut t_cur = t_in;
            for i in 0..2 {
                let u = self.upsample[i];
                // TransConv `k=2, s=2`: `col = b0 [t_cur, 2048]`, `col2im -> b2 [t_cur * 2, 1024]` (residual!)
                push_lin(gpu, &mut chain, b0, u.tc_w, b2, 0, 0, 0, t_cur, 2048, 1024);
                let t_next = t_cur * 2;
                let mut a_c2i = [0u8; 48];
                put_u64(&mut a_c2i, 0, b2);
                put_u64(&mut a_c2i, 8, b0);
                put_u64(&mut a_c2i, 16, 0);
                put_u64(&mut a_c2i, 24, u.tc_b);
                put_u32(&mut a_c2i, 32, t_cur as u32);
                put_u32(&mut a_c2i, 36, 1024);
                put_u32(&mut a_c2i, 40, 2);
                put_u32(&mut a_c2i, 44, 2);
                chain.push(gpu.prepare_qmd(k_col2im, (4, t_next as u32, 1), (256, 1, 1), &a_c2i, None));
                t_cur = t_next;

                // ConvNeXt: `dwconv(b2) -> b1`, update `u.dw_state` from `b2`
                let mut a_dw = [0u8; 48];
                put_u64(&mut a_dw, 0, b1);
                put_u64(&mut a_dw, 8, b2);
                put_u64(&mut a_dw, 16, u.dw_state);
                put_u64(&mut a_dw, 24, u.dw_w);
                put_u64(&mut a_dw, 32, u.dw_b);
                put_u32(&mut a_dw, 40, t_cur as u32);
                put_u32(&mut a_dw, 44, 1024);
                chain.push(gpu.prepare_qmd(k_dw, (4, t_cur as u32, 1), (256, 1, 1), &a_dw, None));

                let tot_dw_st = 6u32 * 1024;
                let mut sa = [0u8; 36];
                put_u64(&mut sa, 0, tmp_st);
                put_u64(&mut sa, 8, u.dw_state);
                put_u64(&mut sa, 16, b2);
                put_u32(&mut sa, 24, t_cur as u32);
                put_u32(&mut sa, 28, 1024);
                put_u32(&mut sa, 32, 6);
                chain.push(gpu.prepare_qmd(k_st_stage, ((tot_dw_st + 255) / 256, 1, 1), (256, 1, 1), &sa, None));

                let mut ca = [0u8; 20];
                put_u64(&mut ca, 0, u.dw_state);
                put_u64(&mut ca, 8, tmp_st);
                put_u32(&mut ca, 16, tot_dw_st);
                chain.push(gpu.prepare_qmd(k_copy, ((tot_dw_st + 255) / 256, 1, 1), (256, 1, 1), &ca, None));

                // LayerNorm `b1 -> b0 [t_cur, 1024]` (`eps = 1e-6`)
                let mut a_ln = [0u8; 44];
                put_u64(&mut a_ln, 0, b0);
                put_u64(&mut a_ln, 8, b1);
                put_u64(&mut a_ln, 16, u.norm_w);
                put_u64(&mut a_ln, 24, u.norm_b);
                put_u32(&mut a_ln, 32, t_cur as u32);
                put_u32(&mut a_ln, 36, 1024);
                put_u32(&mut a_ln, 40, (1e-6f32).to_bits());
                chain.push(gpu.prepare_qmd(k_ln, (t_cur as u32, 1, 1), (256, 1, 1), &a_ln, None));

                // `pwconv1` (`b0 [t_cur, 1024] -> b1 [t_cur, 4096]`) + `GELU(b1)` + `pwconv2 * gamma + b2 -> b2`!
                // Wait: writing `pwconv2` into `b0` with `residual = b2`, then `b0` becomes input to next stage (or copy/swap)!
                // Let's write `pwconv2` -> `b0` (`residual = b2`), and if `i == 0`, swap roles by copying or just tracking `cur_va`!
                push_lin(gpu, &mut chain, b1, u.pw1_w, b0, u.pw1_b, 0, 0, t_cur, 4096, 1024);
                let tot_g = (t_cur * 4096) as u32;
                let mut a_ge = [0u8; 12];
                put_u64(&mut a_ge, 0, b1);
                put_u32(&mut a_ge, 8, tot_g);
                chain.push(gpu.prepare_qmd(k_gelu, ((tot_g + 255) / 256, 1, 1), (256, 1, 1), &a_ge, None));
                push_lin(gpu, &mut chain, b2, u.pw2_w, b1, u.pw2_b, u.gamma, b2, t_cur, 1024, 4096);
            }

            // Stage 5: DAC v2 SnakeBeta Decoder (`b2 [t_in*4, 1024]` -> `audio_out_va [t_in*1920]`)
            // `conv_pre`: `b2 [t_cur, 1024]` -> `b1 [t_cur, 1536]`
            push_causal_conv(
                gpu,
                &mut chain,
                b1,
                self.dac_pre_w,
                b2,
                self.dac_pre_state,
                self.dac_pre_b,
                0,
                t_cur,
                1024,
                1536,
                7,
                1,
            );

            // In each DAC block:
            //   `b1` holds block input `[t_cur, in_ch]`.
            //   `snake(b1)` -> `lin(tc_w, b1) -> b0 (52MB col!)` -> `col2im(b0, tc_carry) -> b2 [t_next, out_ch]` (skip!) + `carry_update`.
            //   Then for each of 3 ResUnits (skip in `b2`, working in `b1` & `b0`):
            //     `copy(b1, b2)` -> `snake(b1, act1)` -> `conv1(b1) -> b0` -> `snake(b0, act2)` -> `conv2(b0, skip=b2) -> b2`!
            //   At end of block, `copy(b1, b2)` (or on block 3 leave in `b1`!).
            for bi in 0..4 {
                let db = self.dac_blks[bi];
                push_snake(gpu, &mut chain, b1, db.snake_a, db.snake_inv_b, t_cur, db.in_ch);
                push_lin(gpu, &mut chain, b0, db.tc_w, b1, 0, 0, 0, t_cur, db.out_ch * db.kernel, db.in_ch);

                let t_next = t_cur * db.stride;
                let mut a_c2i = [0u8; 48];
                put_u64(&mut a_c2i, 0, b2);
                put_u64(&mut a_c2i, 8, b0);
                put_u64(&mut a_c2i, 16, db.tc_carry);
                put_u64(&mut a_c2i, 24, db.tc_b);
                put_u32(&mut a_c2i, 32, t_cur as u32);
                put_u32(&mut a_c2i, 36, db.out_ch as u32);
                put_u32(&mut a_c2i, 40, db.stride as u32);
                put_u32(&mut a_c2i, 44, db.kernel as u32);
                chain.push(gpu.prepare_qmd(
                    k_col2im,
                    (((db.out_ch as u32) + 255) / 256, t_next as u32, 1),
                    (256, 1, 1),
                    &a_c2i,
                    None,
                ));

                let carry_elems = (db.stride * db.out_ch) as u32;
                let mut a_cy = [0u8; 32];
                put_u64(&mut a_cy, 0, db.tc_carry);
                put_u64(&mut a_cy, 8, b0);
                put_u32(&mut a_cy, 16, t_cur as u32);
                put_u32(&mut a_cy, 20, db.out_ch as u32);
                put_u32(&mut a_cy, 24, db.stride as u32);
                put_u32(&mut a_cy, 28, db.kernel as u32);
                chain.push(gpu.prepare_qmd(k_carry, ((carry_elems + 255) / 256, 1, 1), (256, 1, 1), &a_cy, None));

                t_cur = t_next;
                let blk_elems = (t_cur * db.out_ch) as u32;

                for r in 0..3 {
                    let ru = db.res[r];
                    let mut ca = [0u8; 20];
                    put_u64(&mut ca, 0, b1);
                    put_u64(&mut ca, 8, b2);
                    put_u32(&mut ca, 16, blk_elems);
                    chain.push(gpu.prepare_qmd(k_copy, ((blk_elems + 255) / 256, 1, 1), (256, 1, 1), &ca, None));

                    push_snake(gpu, &mut chain, b1, ru.act1_a, ru.act1_inv_b, t_cur, db.out_ch);
                    push_causal_conv(
                        gpu, &mut chain, b0, ru.c1_w, b1, ru.c1_state, ru.c1_b, 0,
                        t_cur, db.out_ch, db.out_ch, 7, ru.dilation,
                    );
                    push_snake(gpu, &mut chain, b0, ru.act2_a, ru.act2_inv_b, t_cur, db.out_ch);
                    // Pointwise `k=1, d=1` conv2 + skip=`b2` -> `b2`! (Wait: writing `b1` with skip=`b2` avoids in-place alias, then swap!)
                    push_causal_conv(
                        gpu, &mut chain, b1, ru.c2_w, b0, 0, ru.c2_b, b2,
                        t_cur, db.out_ch, db.out_ch, 1, 1,
                    );
                    if r < 2 {
                        let mut cb = [0u8; 20];
                        put_u64(&mut cb, 0, b2);
                        put_u64(&mut cb, 8, b1);
                        put_u32(&mut cb, 16, blk_elems);
                        chain.push(gpu.prepare_qmd(k_copy, ((blk_elems + 255) / 256, 1, 1), (256, 1, 1), &cb, None));
                    }
                    // Note: on `r == 2` (last ResUnit of block), output is ALREADY in `b1` ready for the next DAC block!
                }
            }

            // `snake_post(b1, 96)` -> `conv_post(b1 -> b0 [t_cur, 1])` -> `clamp(b0 -> audio_out_va)`!
            push_snake(gpu, &mut chain, b1, self.dac_post_snake_a, self.dac_post_snake_inv_b, t_cur, 96);
            push_causal_conv(
                gpu,
                &mut chain,
                b0,
                self.dac_post_w,
                b1,
                self.dac_post_state,
                self.dac_post_b,
                0,
                t_cur,
                96,
                1,
                7,
                1,
            );

            let n_samp = (t_in * TTS_HOP_SAMPLES) as u32;
            {
                let mut a_cl = [0u8; 20];
                put_u64(&mut a_cl, 0, audio_out_va);
                put_u64(&mut a_cl, 8, b0);
                put_u32(&mut a_cl, 16, n_samp);
                chain.push(gpu.prepare_qmd(k_clamp, ((n_samp + 255) / 256, 1, 1), (256, 1, 1), &a_cl, None));
            }

            Self::execute_qmd_chain(gpu, &chain, 30)?;
            self.stream_pos += t_in;

            let host_audio = unsafe {
                std::slice::from_raw_parts(
                    host_io_buf.as_ptr().add(16384) as *const f32,
                    n_samp as usize,
                )
            };
            all_samples.extend_from_slice(host_audio);
            f_off += t_in;
        }

        Ok(all_samples)
    }

}
