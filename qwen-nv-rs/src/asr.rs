//! 100%-on-RTX-3080 (`sm_86`) `Qwen3-ASR-1.7B` / `0.6B` Speech-to-Text Engine (`Q8_0` / `Q6_K` / `Q4_K_M`).
//!
//! Zero DSP or neural layers execute on the host CPU/M3 Pro:
//! 1. `16 kHz` Reflect-Padded 400-pt Hann STFT + 128-bin Slaney Log10-Mel + Whisper Normalization (`asr_mel_*`)
//! 2. `3x Conv2d(3x3, stride=2, pad=1) + GELU_erf` Subsampler + `conv_out` + Sinusoidal PE (`100 Hz -> 12.5 Hz`)
//! 3. `24`-Layer (`1.7B`) / `18`-Layer (`0.6B`) Bidirectional Online-Softmax Flash-Attention `AuT` Audio Encoder + Projector
//! 4. `28`-Layer `Qwen3` Causal LM Decoder (`qwen_prefill_*` + 1 persistent 68-SM `qwen_persistent_decode_megakernel` launch!)

#![allow(dead_code)]

use std::sync::atomic::{fence, Ordering};
use std::time::{Duration, Instant};

use crate::audio::ASR_SAMPLE_RATE;
use crate::engine::{QwenDecodeCtrl, QwenLayerDesc, QwenModelDesc};
use crate::gguf::{
    decode_1d_f32, decode_tensor_f32_req, extract_quant_weight_bytes, precompute_freqs_cis_f32,
    resolve_asr_model_path, BpeTokenizer, GgufFile, Utf8StreamDecoder,
};
use crate::nv_defs::{put_u32, put_u64, QmdV3};
use crate::nv_hw::{CubinModule, HostMappedBuffer, NvGpu, VramBuffer};

pub const ASR_MAX_AUDIO_SEC: usize = 60;
pub const ASR_MAX_SAMPLES: usize = ASR_MAX_AUDIO_SEC * (ASR_SAMPLE_RATE as usize); // 960,000 samples (60s)
const ASR_CONV_GROUP_CHUNKS: usize = 8; // 8 chunks (8s) per Conv2d sub-pass -> fixed 61.4 MB Conv2d scratch!
const ASR_PREFILL_CHUNK: usize = 64;
const ASR_DEC_MAX_CTX: usize = 1536; // 780 audio tokens (60s) + prompt + 512 generated tokens

const LANG_NAME_MAP: &[(&str, &str)] = &[
    ("zh", "Chinese"),
    ("en", "English"),
    ("yue", "Cantonese"),
    ("ar", "Arabic"),
    ("de", "German"),
    ("fr", "French"),
    ("es", "Spanish"),
    ("pt", "Portuguese"),
    ("id", "Indonesian"),
    ("it", "Italian"),
    ("ko", "Korean"),
    ("ru", "Russian"),
    ("th", "Thai"),
    ("vi", "Vietnamese"),
    ("ja", "Japanese"),
    ("tr", "Turkish"),
    ("hi", "Hindi"),
    ("ms", "Malay"),
    ("nl", "Dutch"),
    ("sv", "Swedish"),
    ("da", "Danish"),
    ("fi", "Finnish"),
    ("pl", "Polish"),
    ("cs", "Czech"),
    ("fil", "Filipino"),
    ("fa", "Persian"),
    ("el", "Greek"),
    ("ro", "Romanian"),
    ("hu", "Hungarian"),
    ("mk", "Macedonian"),
];

fn normalize_lang_hint(lang: &str) -> Option<(&'static str, &'static str)> {
    let trimmed = lang.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("auto") {
        return None;
    }
    for &(code, pub_name) in LANG_NAME_MAP {
        if trimmed.eq_ignore_ascii_case(code) || trimmed.eq_ignore_ascii_case(pub_name) {
            return Some((code, pub_name));
        }
    }
    None
}

#[derive(Clone, Debug, Default)]
pub struct AsrStats {
    pub asr_model: String,
    pub text: String,
    pub detected_language: String,
    pub audio_sec: f64,
    pub n_samples: usize,
    pub mel_frames: usize,
    pub audio_tokens: usize,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub mel_subsample_ms: f64,
    pub encoder_ms: f64,
    pub decoder_prefill_ms: f64,
    pub decoder_decode_ms: f64,
    pub total_ms: f64,
    pub rtf: f64,
    pub speed_x_realtime: f64,
}

impl AsrStats {
    pub fn to_json(&self) -> String {
        let esc = |s: &str| -> String {
            let mut o = String::with_capacity(s.len() + 4);
            for c in s.chars() {
                match c {
                    '"' => o.push_str("\\\""),
                    '\\' => o.push_str("\\\\"),
                    '\n' => o.push_str("\\n"),
                    '\r' => o.push_str("\\r"),
                    '\t' => o.push_str("\\t"),
                    c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
                    c => o.push(c),
                }
            }
            o
        };
        format!(
            "{{\"asr_model\":\"{}\",\"text\":\"{}\",\"language\":\"{}\",\"audio_sec\":{:.3},\
              \"n_samples\":{},\"mel_frames\":{},\"audio_tokens\":{},\"prompt_tokens\":{},\
              \"completion_tokens\":{},\"mel_subsample_ms\":{:.3},\"encoder_ms\":{:.3},\
              \"decoder_prefill_ms\":{:.3},\"decoder_decode_ms\":{:.3},\"total_ms\":{:.3},\
              \"rtf\":{:.4},\"speed_x_realtime\":{:.2}}}",
            esc(&self.asr_model),
            esc(&self.text),
            esc(&self.detected_language),
            self.audio_sec,
            self.n_samples,
            self.mel_frames,
            self.audio_tokens,
            self.prompt_tokens,
            self.completion_tokens,
            self.mel_subsample_ms,
            self.encoder_ms,
            self.decoder_prefill_ms,
            self.decoder_decode_ms,
            self.total_ms,
            self.rtf,
            self.speed_x_realtime
        )
    }
}

#[derive(Clone, Copy)]
struct AsrEncBlockGpu {
    norm_attn_w: u64,
    norm_attn_b: u64,
    attn_q_w: u64,
    attn_q_b: u64,
    attn_q_type: i32,
    attn_k_w: u64,
    attn_k_b: u64,
    attn_k_type: i32,
    attn_v_w: u64,
    attn_v_b: u64,
    attn_v_type: i32,
    attn_out_w: u64,
    attn_out_b: u64,
    attn_out_type: i32,
    norm_ffn_w: u64,
    norm_ffn_b: u64,
    fc1_w: u64,
    fc1_b: u64,
    fc1_type: i32,
    fc2_w: u64,
    fc2_b: u64,
    fc2_type: i32,
}

#[inline]
fn aftercnn_len(mel_len: usize) -> usize {
    let s = |l: usize| (l + 1) / 2;
    s(s(s(mel_len)))
}

fn build_sinusoid_pe(d_model: usize, length: usize) -> Vec<f32> {
    let mut pe = vec![0.0f32; d_model * length];
    let half = d_model / 2;
    if half <= 1 {
        return pe;
    }
    let log_ts = (10000.0f64).ln() / ((half - 1) as f64);
    for p in 0..length {
        for k in 0..half {
            let inv_ts = (-log_ts * (k as f64)).exp();
            let scaled = (p as f64) * inv_ts;
            pe[p * d_model + k] = scaled.sin() as f32;
            pe[p * d_model + half + k] = scaled.cos() as f32;
        }
    }
    pe
}

fn permute_conv2d_oc_ic_9_to_oc_9_ic(src: &[f32], oc_len: usize, ic_len: usize) -> Vec<f32> {
    assert_eq!(src.len(), oc_len * ic_len * 9);
    let mut dst = vec![0.0f32; src.len()];
    for oc in 0..oc_len {
        for ic in 0..ic_len {
            for tap in 0..9 {
                dst[(oc * 9 + tap) * ic_len + ic] = src[(oc * ic_len + ic) * 9 + tap];
            }
        }
    }
    dst
}

struct ByteSlabBuilder {
    bytes: Vec<u8>,
}

impl ByteSlabBuilder {
    fn with_capacity(cap: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(cap),
        }
    }

    fn align256(&mut self) {
        let pad = (256 - (self.bytes.len() % 256)) % 256;
        if pad > 0 {
            self.bytes.resize(self.bytes.len() + pad, 0);
        }
    }

    fn push_bytes(&mut self, raw: &[u8]) -> u64 {
        self.align256();
        let off = self.bytes.len() as u64;
        self.bytes.extend_from_slice(raw);
        off
    }

    fn push_f32(&mut self, vals: &[f32]) -> u64 {
        self.align256();
        let off = self.bytes.len() as u64;
        self.bytes.reserve(vals.len() * 4);
        for &v in vals {
            self.bytes.extend_from_slice(&v.to_le_bytes());
        }
        off
    }
}

pub struct QwenAsrEngine {
    pub asr_ref: String,
    pub model_name: String,
    pub vram_bytes_total: u64,
    tok: BpeTokenizer,

    // Encoder hyperparameters
    enc_n_layers: usize,
    enc_d_model: usize,
    enc_n_heads: usize,
    enc_ffn_dim: usize,
    enc_n_mels: usize,
    enc_ds_hidden: usize,
    enc_out_dim: usize,
    enc_n_window: usize,

    // Decoder hyperparameters
    dec_n_layers: usize,
    dec_hidden: usize,
    dec_inter: usize,
    dec_n_heads: usize,
    dec_n_kv_heads: usize,
    dec_head_dim: usize,
    dec_vocab: usize,
    dec_rms_eps: f32,

    // Special token IDs
    im_start_id: i32,
    im_end_id: i32,
    eos_id: i32,
    newline_id: i32,
    role_system_id: i32,
    role_user_id: i32,
    role_assistant_id: i32,
    audio_start_id: i32,
    audio_end_id: i32,
    audio_pad_id: i32,
    asr_text_id: i32,

    _owned_vram: Vec<VramBuffer>,

    // Frontend & Subsampler GPU pointers
    fe_window_va: u64,
    fe_mel_fb_va: u64,
    fe_dft_cos_sin_va: u64,
    conv0_w_va: u64,
    conv0_b_va: u64,
    conv1_w_va: u64,
    conv1_b_va: u64,
    conv2_w_va: u64,
    conv2_b_va: u64,
    conv_out_w_va: u64,
    conv_out_type: i32,
    pos_emb_va: u64,

    // Encoder blocks & post-projector GPU pointers
    enc_blocks: Vec<AsrEncBlockGpu>,
    ln_post_w_va: u64,
    ln_post_b_va: u64,
    proj1_w_va: u64,
    proj1_b_va: u64,
    proj1_type: i32,
    proj2_w_va: u64,
    proj2_b_va: u64,
    proj2_type: i32,

    // Decoder megakernel descriptors & pointers
    dec_emb_va: u64,
    dec_emb_type: i32,
    dec_freqs_va: u64,
    dec_layers: Vec<QwenLayerDesc>,
    dec_model_desc_gpu: VramBuffer,

    // Reusable VRAM workspaces
    ws_pcm: VramBuffer,
    ws_max_int: VramBuffer,
    ws_log_mel: VramBuffer,
    ws_mel_chunks: VramBuffer,
    ws_conv_a: VramBuffer,
    ws_conv_b: VramBuffer,
    ws_conv_out_pad: VramBuffer,
    ws_enc_x: VramBuffer,
    ws_enc_x2: VramBuffer,
    ws_enc_q: VramBuffer,
    ws_enc_k: VramBuffer,
    ws_enc_v: VramBuffer,
    ws_enc_ffn: VramBuffer,
    ws_enc_proj_out: VramBuffer,

    ws_dec_prefill_emb: VramBuffer,
    ws_dec_x_res: VramBuffer,
    ws_dec_x_res2: VramBuffer,
    ws_dec_xq: VramBuffer,
    ws_dec_xds: VramBuffer,
    ws_dec_qkv: VramBuffer,
    ws_dec_v_sep: VramBuffer,
    ws_dec_q_roped: VramBuffer,
    ws_dec_attn_out: VramBuffer,
    ws_dec_gate_up: VramBuffer,
    ws_dec_bar_state: VramBuffer,

    host_prompt_buf: HostMappedBuffer,
    dec_ctrl_buf: HostMappedBuffer,
}

impl QwenAsrEngine {
    pub fn load(gpu: &mut NvGpu, asr_ref: &str) -> Result<Self, String> {
        let t0 = Instant::now();
        let gguf_path = resolve_asr_model_path(asr_ref)?;
        let gf = GgufFile::open(&gguf_path)?;
        let tok = BpeTokenizer::from_gguf(&gf)?;

        let get_u = |k: &str| -> Result<usize, String> {
            gf.kv
                .get(k)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .ok_or_else(|| format!("Missing Qwen3-ASR GGUF key '{}'", k))
        };
        let get_f = |k: &str, def: f32| -> f32 {
            gf.kv.get(k).and_then(|v| v.as_f32()).unwrap_or(def)
        };

        let model_name = gf
            .kv
            .get("general.name")
            .and_then(|v| v.as_str())
            .unwrap_or("Qwen3-ASR-1.7B")
            .to_string();

        let enc_n_layers = get_u("stt.qwen3_asr.encoder.n_layers")?;
        let enc_d_model = get_u("stt.qwen3_asr.encoder.d_model")?;
        let enc_n_heads = get_u("stt.qwen3_asr.encoder.n_heads")?;
        let enc_ffn_dim = get_u("stt.qwen3_asr.encoder.ffn_dim")?;
        let enc_n_mels = get_u("stt.qwen3_asr.encoder.num_mel_bins")?;
        let enc_ds_hidden = get_u("stt.qwen3_asr.encoder.downsample_hidden")?;
        let enc_out_dim = get_u("stt.qwen3_asr.encoder.output_dim")?;
        let enc_n_window = get_u("stt.qwen3_asr.encoder.n_window")?;

        let dec_n_layers = get_u("stt.qwen3_asr.decoder.n_layers")?;
        let dec_hidden = get_u("stt.qwen3_asr.decoder.hidden_size")?;
        let dec_inter = get_u("stt.qwen3_asr.decoder.intermediate_size")?;
        let dec_n_heads = get_u("stt.qwen3_asr.decoder.n_heads")?;
        let dec_n_kv_heads = get_u("stt.qwen3_asr.decoder.n_kv_heads")?;
        let dec_head_dim = get_u("stt.qwen3_asr.decoder.head_dim")?;
        let dec_vocab = get_u("stt.qwen3_asr.decoder.vocab_size")?;
        let dec_rms_eps = get_f("stt.qwen3_asr.decoder.rms_norm_eps", 1e-6);
        let dec_rope_theta = get_f("stt.qwen3_asr.decoder.rope_theta", 1_000_000.0);

        if enc_d_model / enc_n_heads != 64 {
            return Err(format!(
                "Expected encoder head_dim=64, got {}/{}",
                enc_d_model, enc_n_heads
            ));
        }
        if enc_out_dim != dec_hidden {
            return Err(format!(
                "Encoder output_dim ({}) != decoder hidden_size ({})",
                enc_out_dim, dec_hidden
            ));
        }

        let audio_pad_id = get_u("stt.qwen3_asr.audio_token_id")? as i32;
        let audio_start_id = get_u("stt.qwen3_asr.audio_start_token_id")? as i32;
        let audio_end_id = get_u("stt.qwen3_asr.audio_end_token_id")? as i32;
        let eos_id = gf
            .kv
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u64())
            .unwrap_or(151645) as i32;

        let req_tok = |s: &str| -> Result<i32, String> {
            tok.find_token_id(s)
                .ok_or_else(|| format!("Qwen3-ASR tokenizer missing piece {:?}", s))
        };
        let im_start_id = req_tok("<|im_start|>")?;
        let im_end_id = req_tok("<|im_end|>")?;
        let newline_id = req_tok("\n")?;
        let role_system_id = req_tok("system")?;
        let role_user_id = req_tok("user")?;
        let role_assistant_id = req_tok("assistant")?;
        let asr_text_id = req_tok("<asr_text>")?;

        let mut owned_vram = Vec::<VramBuffer>::new();
        let mut total_vram = 0u64;
        let mut alloc_track = |gpu: &mut NvGpu, sz: u64| -> Result<VramBuffer, String> {
            let b = gpu.alloc_vram(sz.max(256))?;
            total_vram += b.size;
            Ok(b)
        };

        // 1. Build F32 small-tensor & conv2d arena (`~20 MB`)
        let mut f32_arena = ByteSlabBuilder::with_capacity(24 << 20);
        let dec_f32 = |name: &str| -> Result<Vec<f32>, String> { Ok(decode_tensor_f32_req(&gf, name)?.0) };

        let fe_win = dec_f32("frontend.window")?;
        let fe_fb = dec_f32("frontend.mel_filterbank")?;
        if fe_win.len() != 400 || fe_fb.len() != 128 * 201 {
            return Err(format!(
                "Unexpected frontend shapes: window={}, mel_fb={}",
                fe_win.len(),
                fe_fb.len()
            ));
        }
        let mut dft_cs = vec![0.0f32; 201 * 800];
        for k in 0..201usize {
            for n in 0..400usize {
                let ang = 2.0 * std::f64::consts::PI * (k as f64) * (n as f64) / 400.0;
                dft_cs[k * 800 + n] = ang.cos() as f32;
                dft_cs[k * 800 + 400 + n] = (-ang.sin()) as f32;
            }
        }
        let t_per_chunk = aftercnn_len(enc_n_window * 2);
        let pos_emb = build_sinusoid_pe(enc_d_model, t_per_chunk);

        let fe_win_rel = f32_arena.push_f32(&fe_win);
        let fe_fb_rel = f32_arena.push_f32(&fe_fb);
        let dft_cs_rel = f32_arena.push_f32(&dft_cs);
        let pos_emb_rel = f32_arena.push_f32(&pos_emb);

        let c0_w_raw = dec_f32("enc.conv.0.weight")?;
        let c0_w = permute_conv2d_oc_ic_9_to_oc_9_ic(&c0_w_raw, enc_ds_hidden, 1);
        let c0_b = dec_f32("enc.conv.0.bias")?;
        let c1_w_raw = dec_f32("enc.conv.1.weight")?;
        let c1_w = permute_conv2d_oc_ic_9_to_oc_9_ic(&c1_w_raw, enc_ds_hidden, enc_ds_hidden);
        let c1_b = dec_f32("enc.conv.1.bias")?;
        let c2_w_raw = dec_f32("enc.conv.2.weight")?;
        let c2_w = permute_conv2d_oc_ic_9_to_oc_9_ic(&c2_w_raw, enc_ds_hidden, enc_ds_hidden);
        let c2_b = dec_f32("enc.conv.2.bias")?;

        let c0_w_rel = f32_arena.push_f32(&c0_w);
        let c0_b_rel = f32_arena.push_f32(&c0_b);
        let c1_w_rel = f32_arena.push_f32(&c1_w);
        let c1_b_rel = f32_arena.push_f32(&c1_b);
        let c2_w_rel = f32_arena.push_f32(&c2_w);
        let c2_b_rel = f32_arena.push_f32(&c2_b);

        let ln_post_w_rel = f32_arena.push_f32(&dec_f32("enc.ln_post.weight")?);
        let ln_post_b_rel = f32_arena.push_f32(&dec_f32("enc.ln_post.bias")?);
        let proj1_b_rel = f32_arena.push_f32(&dec_f32("enc.proj1.bias")?);
        let proj2_b_rel = f32_arena.push_f32(&dec_f32("enc.proj2.bias")?);

        let dec_out_norm_rel =
            f32_arena.push_f32(&dec_f32("dec.output_norm.weight")?);
        let dec_freqs = precompute_freqs_cis_f32(dec_head_dim, ASR_DEC_MAX_CTX, dec_rope_theta);
        let dec_freqs_rel = f32_arena.push_f32(&dec_freqs);

        // Collect per-layer 1D F32 norms/biases into `f32_arena` first so base VA resolves cleanly
        struct EncRelF32 {
            na_w: u64,
            na_b: u64,
            qb: u64,
            kb: u64,
            vb: u64,
            ob: u64,
            nf_w: u64,
            nf_b: u64,
            fc1_b: u64,
            fc2_b: u64,
        }
        let mut enc_rels = Vec::with_capacity(enc_n_layers);
        for i in 0..enc_n_layers {
            let p = format!("enc.blocks.{}", i);
            enc_rels.push(EncRelF32 {
                na_w: f32_arena.push_f32(&dec_f32(&format!("{}.norm_attn.weight", p))?),
                na_b: f32_arena.push_f32(&dec_f32(&format!("{}.norm_attn.bias", p))?),
                qb: f32_arena.push_f32(&dec_f32(&format!("{}.attn.q.bias", p))?),
                kb: f32_arena.push_f32(&dec_f32(&format!("{}.attn.k.bias", p))?),
                vb: f32_arena.push_f32(&dec_f32(&format!("{}.attn.v.bias", p))?),
                ob: f32_arena.push_f32(&dec_f32(&format!("{}.attn.out.bias", p))?),
                nf_w: f32_arena.push_f32(&dec_f32(&format!("{}.norm_ffn.weight", p))?),
                nf_b: f32_arena.push_f32(&dec_f32(&format!("{}.norm_ffn.bias", p))?),
                fc1_b: f32_arena.push_f32(&dec_f32(&format!("{}.ffn.fc1.bias", p))?),
                fc2_b: f32_arena.push_f32(&dec_f32(&format!("{}.ffn.fc2.bias", p))?),
            });
        }

        struct DecRelF32 {
            na_w: u64,
            nf_w: u64,
            qn_w: u64,
            kn_w: u64,
        }
        let mut dec_rels = Vec::with_capacity(dec_n_layers);
        for i in 0..dec_n_layers {
            let p = format!("dec.blocks.{}", i);
            let na = decode_1d_f32(&gf, &format!("{}.norm_attn.weight", p))?
                .ok_or_else(|| format!("Missing {}.norm_attn.weight", p))?;
            let nf = decode_1d_f32(&gf, &format!("{}.norm_ffn.weight", p))?
                .ok_or_else(|| format!("Missing {}.norm_ffn.weight", p))?;
            let qn = decode_1d_f32(&gf, &format!("{}.attn.q_norm.weight", p))?
                .ok_or_else(|| format!("Missing {}.attn.q_norm.weight", p))?;
            let kn = decode_1d_f32(&gf, &format!("{}.attn.k_norm.weight", p))?
                .ok_or_else(|| format!("Missing {}.attn.k_norm.weight", p))?;
            dec_rels.push(DecRelF32 {
                na_w: f32_arena.push_f32(&na),
                nf_w: f32_arena.push_f32(&nf),
                qn_w: f32_arena.push_f32(&qn),
                kn_w: f32_arena.push_f32(&kn),
            });
        }

        let f32_buf = alloc_track(gpu, f32_arena.bytes.len() as u64)?;
        gpu.upload_to_vram(&f32_buf, 0, &f32_arena.bytes)?;
        let f32_base = f32_buf.va_addr;
        owned_vram.push(f32_buf);

        // Decoder FP16 KV Cache (`dec_n_layers * 2 * dec_n_kv_heads * ASR_DEC_MAX_CTX * dec_head_dim * 2`)
        let kv_layer_bytes =
            ((2 * dec_n_kv_heads * ASR_DEC_MAX_CTX * dec_head_dim * 2) as u64 + 255) & !255;
        let dec_kv_slab = alloc_track(gpu, kv_layer_bytes * (dec_n_layers as u64))?;
        let dec_kv_base = dec_kv_slab.va_addr;
        owned_vram.push(dec_kv_slab);

        // 2. Upload Quantized Encoder + Decoder Weights in 64 MB Hugepage-Friendly Slabs
        const SLAB_CAP: usize = 64 << 20;
        let mut cur_slab = ByteSlabBuilder::with_capacity(SLAB_CAP);
        let mut cur_slab_buf: Option<VramBuffer> = None;

        let flush_slab = |gpu: &mut NvGpu,
                          slab: &mut ByteSlabBuilder,
                          buf_opt: &mut Option<VramBuffer>,
                          owned: &mut Vec<VramBuffer>|
         -> Result<(), String> {
            if let Some(b) = buf_opt.take() {
                gpu.upload_to_vram(&b, 0, &slab.bytes)?;
                owned.push(b);
                slab.bytes.clear();
            }
            Ok(())
        };

        let mut stage_quant = |gpu: &mut NvGpu, raw: &[u8]| -> Result<u64, String> {
            if raw.len() > SLAB_CAP / 2 {
                let b = alloc_track(gpu, raw.len() as u64)?;
                gpu.upload_to_vram(&b, 0, raw)?;
                let va = b.va_addr;
                owned_vram.push(b);
                return Ok(va);
            }
            if cur_slab_buf.is_none() || cur_slab.bytes.len() + raw.len() + 256 > SLAB_CAP {
                flush_slab(gpu, &mut cur_slab, &mut cur_slab_buf, &mut owned_vram)?;
                cur_slab_buf = Some(alloc_track(gpu, SLAB_CAP as u64)?);
            }
            let base = cur_slab_buf.as_ref().unwrap().va_addr;
            let rel = cur_slab.push_bytes(raw);
            Ok(base + rel)
        };

        let (co_cow, conv_out_type, _, _) = extract_quant_weight_bytes(&gf, "enc.conv_out.weight")?;
        let conv_out_w_va = stage_quant(gpu, &co_cow)?;
        let (p1_cow, proj1_type, _, _) = extract_quant_weight_bytes(&gf, "enc.proj1.weight")?;
        let proj1_w_va = stage_quant(gpu, &p1_cow)?;
        let (p2_cow, proj2_type, _, _) = extract_quant_weight_bytes(&gf, "enc.proj2.weight")?;
        let proj2_w_va = stage_quant(gpu, &p2_cow)?;

        let mut enc_blocks = Vec::with_capacity(enc_n_layers);
        for i in 0..enc_n_layers {
            let p = format!("enc.blocks.{}", i);
            let (qw, qt, _, _) = extract_quant_weight_bytes(&gf, &format!("{}.attn.q.weight", p))?;
            let (kw, kt, _, _) = extract_quant_weight_bytes(&gf, &format!("{}.attn.k.weight", p))?;
            let (vw, vt, _, _) = extract_quant_weight_bytes(&gf, &format!("{}.attn.v.weight", p))?;
            let (ow, ot, _, _) = extract_quant_weight_bytes(&gf, &format!("{}.attn.out.weight", p))?;
            let (f1w, f1t, _, _) = extract_quant_weight_bytes(&gf, &format!("{}.ffn.fc1.weight", p))?;
            let (f2w, f2t, _, _) = extract_quant_weight_bytes(&gf, &format!("{}.ffn.fc2.weight", p))?;
            let r = &enc_rels[i];

            enc_blocks.push(AsrEncBlockGpu {
                norm_attn_w: f32_base + r.na_w,
                norm_attn_b: f32_base + r.na_b,
                attn_q_w: stage_quant(gpu, &qw)?,
                attn_q_b: f32_base + r.qb,
                attn_q_type: qt,
                attn_k_w: stage_quant(gpu, &kw)?,
                attn_k_b: f32_base + r.kb,
                attn_k_type: kt,
                attn_v_w: stage_quant(gpu, &vw)?,
                attn_v_b: f32_base + r.vb,
                attn_v_type: vt,
                attn_out_w: stage_quant(gpu, &ow)?,
                attn_out_b: f32_base + r.ob,
                attn_out_type: ot,
                norm_ffn_w: f32_base + r.nf_w,
                norm_ffn_b: f32_base + r.nf_b,
                fc1_w: stage_quant(gpu, &f1w)?,
                fc1_b: f32_base + r.fc1_b,
                fc1_type: f1t,
                fc2_w: stage_quant(gpu, &f2w)?,
                fc2_b: f32_base + r.fc2_b,
                fc2_type: f2t,
            });
        }

        // Tied Decoder Embedding / LM Head
        let (emb_cow, dec_emb_type, _, _) = extract_quant_weight_bytes(&gf, "dec.token_embd.weight")?;
        let dec_emb_va = stage_quant(gpu, &emb_cow)?;

        let mut dec_layers = Vec::with_capacity(dec_n_layers);
        for i in 0..dec_n_layers {
            let p = format!("dec.blocks.{}", i);
            let (qw, qt, _, _) = extract_quant_weight_bytes(&gf, &format!("{}.attn.q.weight", p))?;
            let (kw, kt, _, _) = extract_quant_weight_bytes(&gf, &format!("{}.attn.k.weight", p))?;
            let (vw, vt, _, _) = extract_quant_weight_bytes(&gf, &format!("{}.attn.v.weight", p))?;
            let (ow, ot, _, _) = extract_quant_weight_bytes(&gf, &format!("{}.attn.o.weight", p))?;
            let (gw, gt, _, _) = extract_quant_weight_bytes(&gf, &format!("{}.ffn.gate.weight", p))?;
            let (uw, ut, _, _) = extract_quant_weight_bytes(&gf, &format!("{}.ffn.up.weight", p))?;
            let (dw, dt, _, _) = extract_quant_weight_bytes(&gf, &format!("{}.ffn.down.weight", p))?;

            if qt != kt || gt != ut {
                return Err(format!("Mismatched q/k or gate/up quant types at ASR dec layer {}", i));
            }
            let (w_qkv_or_qk, w_v_sep, v_type_desc, split_v_flag) = if vt == qt {
                let mut qkv = Vec::with_capacity(qw.len() + kw.len() + vw.len());
                qkv.extend_from_slice(&qw);
                qkv.extend_from_slice(&kw);
                qkv.extend_from_slice(&vw);
                (stage_quant(gpu, &qkv)?, 0u64, vt, 0i32)
            } else {
                let mut qk = Vec::with_capacity(qw.len() + kw.len());
                qk.extend_from_slice(&qw);
                qk.extend_from_slice(&kw);
                (stage_quant(gpu, &qk)?, stage_quant(gpu, &vw)?, vt, 1i32)
            };

            let mut gu = Vec::with_capacity(gw.len() + uw.len());
            gu.extend_from_slice(&gw);
            gu.extend_from_slice(&uw);
            let w_gate_up = stage_quant(gpu, &gu)?;
            let w_o = stage_quant(gpu, &ow)?;
            let w_down = stage_quant(gpu, &dw)?;
            let r = &dec_rels[i];

            dec_layers.push(QwenLayerDesc {
                attn_norm_w: f32_base + r.na_w,
                ffn_norm_w: f32_base + r.nf_w,
                q_norm_w: f32_base + r.qn_w,
                k_norm_w: f32_base + r.kn_w,
                w_qkv_or_qk,
                b_qkv_or_qk: 0,
                w_v: w_v_sep,
                b_v: 0,
                w_o,
                b_o: 0,
                w_gate_up,
                w_up_sep: 0,
                w_down,
                b_down: 0,
                kv_cache: dec_kv_base + (i as u64) * kv_layer_bytes,
                qkv_type: qt,
                v_type: v_type_desc,
                o_type: ot,
                gate_up_type: gt,
                up_sep_type: 0,
                down_type: dt,
                split_v: split_v_flag,
                split_gate_up: 0,
                lora_ptr: 0,
                _pad_lora: 0,
            });
        }
        flush_slab(gpu, &mut cur_slab, &mut cur_slab_buf, &mut owned_vram)?;

        let dec_layers_bytes = unsafe {
            std::slice::from_raw_parts(
                dec_layers.as_ptr() as *const u8,
                dec_layers.len() * std::mem::size_of::<QwenLayerDesc>(),
            )
        };
        let dec_layers_gpu = alloc_track(gpu, dec_layers_bytes.len() as u64)?;
        gpu.upload_to_vram(&dec_layers_gpu, 0, dec_layers_bytes)?;
        let dec_layers_va = dec_layers_gpu.va_addr;
        owned_vram.push(dec_layers_gpu);

        // 3. Allocate Reusable Encoder + Decoder Scratch Buffers
        let mel_per_chunk = enc_n_window * 2; // 100
        let max_mel_frames = ASR_MAX_SAMPLES / 160; // 6000
        let max_chunks = (max_mel_frames + mel_per_chunk - 1) / mel_per_chunk; // 60
        let max_t_enc_pad = max_chunks * t_per_chunk; // 780

        let ws_pcm = alloc_track(gpu, (ASR_MAX_SAMPLES * 4) as u64)?;
        let ws_max_int = alloc_track(gpu, 256)?;
        let ws_log_mel = alloc_track(gpu, (enc_n_mels * max_mel_frames * 4) as u64)?;
        let ws_mel_chunks = alloc_track(gpu, (max_chunks * enc_n_mels * mel_per_chunk * 4) as u64)?;

        // Conv2d group buffers (`ASR_CONV_GROUP_CHUNKS = 8`)
        let h1 = (enc_n_mels + 1) / 2; // 64
        let w1 = (mel_per_chunk + 1) / 2; // 50
        let h2 = (h1 + 1) / 2; // 32
        let w2 = (w1 + 1) / 2; // 25
        let ws_conv_a = alloc_track(gpu, (ASR_CONV_GROUP_CHUNKS * enc_ds_hidden * h1 * w1 * 4) as u64)?;
        let ws_conv_b = alloc_track(gpu, (ASR_CONV_GROUP_CHUNKS * enc_ds_hidden * h2 * w2 * 4) as u64)?;
        let ws_conv_out_pad = alloc_track(gpu, (max_t_enc_pad * enc_d_model * 4) as u64)?;

        let ws_enc_x = alloc_track(gpu, (max_t_enc_pad * enc_d_model * 4) as u64)?;
        let ws_enc_x2 = alloc_track(gpu, (max_t_enc_pad * enc_d_model * 4) as u64)?;
        let ws_enc_q = alloc_track(gpu, (max_t_enc_pad * enc_d_model * 4) as u64)?;
        let ws_enc_k = alloc_track(gpu, (max_t_enc_pad * enc_d_model * 4) as u64)?;
        let ws_enc_v = alloc_track(gpu, (max_t_enc_pad * enc_d_model * 4) as u64)?;
        let ws_enc_ffn = alloc_track(gpu, (max_t_enc_pad * enc_ffn_dim * 4) as u64)?;
        let ws_enc_proj_out = alloc_track(gpu, (max_t_enc_pad * enc_out_dim * 4) as u64)?;

        // Shared Q8_1 scratch (`ws_dec_xq` / `ws_dec_xds`) sized for MAX of encoder (`max_t_enc_pad * max(7680, enc_ffn_dim)`)
        // and decoder (`ASR_PREFILL_CHUNK * dec_inter`)!
        let max_q8_elems = (max_t_enc_pad * (enc_ds_hidden * 16).max(enc_ffn_dim))
            .max(ASR_PREFILL_CHUNK * (2 * dec_inter));
        let ws_dec_xq = alloc_track(gpu, max_q8_elems as u64)?;
        let ws_dec_xds = alloc_track(gpu, ((max_q8_elems / 32) * 8) as u64)?;

        let q_dim = dec_n_heads * dec_head_dim;
        let kv_dim = dec_n_kv_heads * dec_head_dim;
        let ws_dec_prefill_emb = alloc_track(gpu, (ASR_DEC_MAX_CTX * dec_hidden * 4) as u64)?;
        let ws_dec_x_res = alloc_track(gpu, (ASR_PREFILL_CHUNK * dec_hidden * 4) as u64)?;
        let ws_dec_x_res2 = alloc_track(gpu, (ASR_PREFILL_CHUNK * dec_hidden * 4) as u64)?;
        let ws_dec_qkv = alloc_track(gpu, (ASR_PREFILL_CHUNK * (q_dim + 2 * kv_dim) * 4) as u64)?;
        let ws_dec_v_sep = alloc_track(gpu, (ASR_PREFILL_CHUNK * kv_dim * 4) as u64)?;
        let ws_dec_q_roped = alloc_track(gpu, (ASR_PREFILL_CHUNK * q_dim * 4) as u64)?;
        let ws_dec_attn_out = alloc_track(gpu, (ASR_PREFILL_CHUNK * q_dim * 4) as u64)?;
        let ws_dec_gate_up = alloc_track(gpu, (ASR_PREFILL_CHUNK * 2 * dec_inter * 4) as u64)?;
        let ws_dec_cta_val = alloc_track(gpu, 4096)?;
        let ws_dec_cta_idx = alloc_track(gpu, 4096)?;
        let cta_val_va = ws_dec_cta_val.va_addr;
        let cta_idx_va = ws_dec_cta_idx.va_addr;
        owned_vram.push(ws_dec_cta_val);
        owned_vram.push(ws_dec_cta_idx);
        let ws_dec_bar_state = alloc_track(gpu, 4096)?;
        gpu.zero_vram(&ws_dec_bar_state)?;

        let dec_desc = QwenModelDesc {
            layers_ptr: dec_layers_va,
            emb_w: dec_emb_va,
            output_norm_w: f32_base + dec_out_norm_rel,
            lm_head_w: dec_emb_va,
            freqs_cis: f32_base + dec_freqs_rel,
            ws_x_res: ws_dec_x_res.va_addr,
            ws_x_res2: ws_dec_x_res2.va_addr,
            ws_xq: ws_dec_xq.va_addr,
            ws_xds: ws_dec_xds.va_addr,
            ws_qkv: ws_dec_qkv.va_addr,
            ws_q_roped: ws_dec_q_roped.va_addr,
            ws_gate_up: ws_dec_gate_up.va_addr,
            ws_cta_val: cta_val_va,
            ws_cta_idx: cta_idx_va,
            ws_bar_state: ws_dec_bar_state.va_addr,
            num_layers: dec_n_layers as i32,
            dim: dec_hidden as i32,
            hidden_dim: dec_inter as i32,
            n_heads: dec_n_heads as i32,
            n_kv_heads: dec_n_kv_heads as i32,
            head_dim: dec_head_dim as i32,
            rope_dim: dec_head_dim as i32,
            vocab_size: dec_vocab as i32,
            max_ctx: ASR_DEC_MAX_CTX as i32,
            has_qk_norm: 1,
            emb_type: dec_emb_type,
            lm_head_type: dec_emb_type,
            norm_eps: dec_rms_eps,
            inv_sqrt_head_dim: 1.0 / (dec_head_dim as f32).sqrt(),
            stop_id0: im_end_id,
            stop_id1: eos_id,
            stop_id2: im_end_id,
            stop_id3: eos_id,
            ws_ext_embed: 0,
            ws_hidden_last: 0,
            ws_logits_out: 0,
            _pad_model: 0,
        };
        let dec_desc_bytes = unsafe {
            std::slice::from_raw_parts(
                &dec_desc as *const QwenModelDesc as *const u8,
                std::mem::size_of::<QwenModelDesc>(),
            )
        };
        let dec_model_desc_gpu = alloc_track(gpu, 256)?;
        gpu.upload_to_vram(&dec_model_desc_gpu, 0, dec_desc_bytes)?;

        let host_prompt_buf = gpu.alloc_host_mapped((ASR_DEC_MAX_CTX * 4).max(8192))?;
        let dec_ctrl_buf = gpu.alloc_host_mapped(std::mem::size_of::<QwenDecodeCtrl>())?;

        let vram_mb = (total_vram as f64) / (1024.0 * 1024.0);
        eprintln!(
            "[qwen-nv asr] Loaded '{}' ({:.1} MB VRAM, {}L AuT encoder + {}L Qwen3 megakernel, 100% RTX 3080 sm_86) in {:.2}s",
            model_name,
            vram_mb,
            enc_n_layers,
            dec_n_layers,
            t0.elapsed().as_secs_f64()
        );

        Ok(Self {
            asr_ref: asr_ref.to_string(),
            model_name,
            vram_bytes_total: total_vram,
            tok,
            enc_n_layers,
            enc_d_model,
            enc_n_heads,
            enc_ffn_dim,
            enc_n_mels,
            enc_ds_hidden,
            enc_out_dim,
            enc_n_window,
            dec_n_layers,
            dec_hidden,
            dec_inter,
            dec_n_heads,
            dec_n_kv_heads,
            dec_head_dim,
            dec_vocab,
            dec_rms_eps,
            im_start_id,
            im_end_id,
            eos_id,
            newline_id,
            role_system_id,
            role_user_id,
            role_assistant_id,
            audio_start_id,
            audio_end_id,
            audio_pad_id,
            asr_text_id,
            _owned_vram: owned_vram,
            fe_window_va: f32_base + fe_win_rel,
            fe_mel_fb_va: f32_base + fe_fb_rel,
            fe_dft_cos_sin_va: f32_base + dft_cs_rel,
            conv0_w_va: f32_base + c0_w_rel,
            conv0_b_va: f32_base + c0_b_rel,
            conv1_w_va: f32_base + c1_w_rel,
            conv1_b_va: f32_base + c1_b_rel,
            conv2_w_va: f32_base + c2_w_rel,
            conv2_b_va: f32_base + c2_b_rel,
            conv_out_w_va,
            conv_out_type,
            pos_emb_va: f32_base + pos_emb_rel,
            enc_blocks,
            ln_post_w_va: f32_base + ln_post_w_rel,
            ln_post_b_va: f32_base + ln_post_b_rel,
            proj1_w_va,
            proj1_b_va: f32_base + proj1_b_rel,
            proj1_type,
            proj2_w_va,
            proj2_b_va: f32_base + proj2_b_rel,
            proj2_type,
            dec_emb_va,
            dec_emb_type,
            dec_freqs_va: f32_base + dec_freqs_rel,
            dec_layers,
            dec_model_desc_gpu,
            ws_pcm,
            ws_max_int,
            ws_log_mel,
            ws_mel_chunks,
            ws_conv_a,
            ws_conv_b,
            ws_conv_out_pad,
            ws_enc_x,
            ws_enc_x2,
            ws_enc_q,
            ws_enc_k,
            ws_enc_v,
            ws_enc_ffn,
            ws_enc_proj_out,
            ws_dec_prefill_emb,
            ws_dec_x_res,
            ws_dec_x_res2,
            ws_dec_xq,
            ws_dec_xds,
            ws_dec_qkv,
            ws_dec_v_sep,
            ws_dec_q_roped,
            ws_dec_attn_out,
            ws_dec_gate_up,
            ws_dec_bar_state,
            host_prompt_buf,
            dec_ctrl_buf,
        })
    }

    fn execute_qmd_chain(gpu: &mut NvGpu, chain: &[(u64, usize)], timeout_secs: u64) -> Result<(), String> {
        gpu.timeline_value += 1;
        let target_sig = gpu.timeline_value;
        let sig_va = gpu.timeline_buf.va_addr;
        {
            let (_, last_off) = *chain.last().unwrap();
            let buf = gpu.kernargs_buf_mut_slice_at(last_off);
            let mut tail_qmd = QmdV3 {
                bytes: buf[..256].try_into().unwrap(),
            };
            tail_qmd.set_release0(sig_va, target_sig);
            buf[..256].copy_from_slice(&tail_qmd.bytes);
        }
        for i in 0..(chain.len() - 1) {
            gpu.link_dependent_qmd(chain[i].1, chain[i + 1].0);
        }
        gpu.launch_qmd_chain(chain[0].0, true)?;
        gpu.wait_timeline(target_sig, Duration::from_secs(timeout_secs))
    }

    fn run_dec_prefill_chunk(
        &self,
        gpu: &mut NvGpu,
        module: &CubinModule,
        start_pos: usize,
        t_len: usize,
    ) -> Result<(), String> {
        let k_copy = &module.kernels["tts_copy_f32"];
        let k_rms = &module.kernels["qwen_prefill_rmsnorm_q8_1"];
        let k_gemm = &module.kernels["qwen_prefill_gemm"];
        let k_qknorm = &module.kernels["qwen_prefill_qknorm_rope_kv"];
        let k_flash = &module.kernels["qwen_prefill_flash_gqa"];
        let k_quant = &module.kernels["qwen_prefill_quant_q8_1"];
        let k_swiglu = &module.kernels["qwen_prefill_swiglu_q8_1"];

        let dim = self.dec_hidden as u32;
        let hidden_dim = self.dec_inter as u32;
        let n_heads = self.dec_n_heads as u32;
        let n_kv_heads = self.dec_n_kv_heads as u32;
        let head_dim = self.dec_head_dim as u32;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let m_u = t_len as u32;
        let grid_m = (m_u + 7) / 8;
        let eps_bits = self.dec_rms_eps.to_bits();
        let scale_bits = (1.0f32 / (self.dec_head_dim as f32).sqrt()).to_bits();
        let max_ctx_u = ASR_DEC_MAX_CTX as u32;

        let mut chain = Vec::with_capacity(2 + self.dec_n_layers * 8);

        // Copy `ws_dec_prefill_emb[start_pos .. start_pos + t_len]` -> `ws_dec_x_res`
        let copy_elems = (t_len * self.dec_hidden) as u32;
        let src_va = self.ws_dec_prefill_emb.va_addr + (start_pos * self.dec_hidden * 4) as u64;
        let mut a_cp = [0u8; 20];
        put_u64(&mut a_cp, 0, self.ws_dec_x_res.va_addr);
        put_u64(&mut a_cp, 8, src_va);
        put_u32(&mut a_cp, 16, copy_elems);
        chain.push(gpu.prepare_qmd(k_copy, ((copy_elems + 255) / 256, 1, 1), (256, 1, 1), &a_cp, None));

        for l in 0..self.dec_n_layers {
            let ld = self.dec_layers[l];
            let mut a_rms1 = [0u8; 40];
            put_u64(&mut a_rms1, 0, self.ws_dec_xq.va_addr);
            put_u64(&mut a_rms1, 8, self.ws_dec_xds.va_addr);
            put_u64(&mut a_rms1, 16, self.ws_dec_x_res.va_addr);
            put_u64(&mut a_rms1, 24, ld.attn_norm_w);
            put_u32(&mut a_rms1, 32, dim);
            put_u32(&mut a_rms1, 36, eps_bits);
            chain.push(gpu.prepare_qmd(k_rms, (m_u, 1, 1), (256, 1, 1), &a_rms1, None));

            let qkv_n = if ld.split_v == 0 { q_dim + 2 * kv_dim } else { q_dim + kv_dim };
            let mut a_qk = [0u8; 64];
            put_u64(&mut a_qk, 0, self.ws_dec_qkv.va_addr);
            put_u64(&mut a_qk, 8, ld.w_qkv_or_qk);
            put_u64(&mut a_qk, 16, self.ws_dec_xq.va_addr);
            put_u64(&mut a_qk, 24, self.ws_dec_xds.va_addr);
            put_u64(&mut a_qk, 32, 0);
            put_u64(&mut a_qk, 40, 0);
            put_u32(&mut a_qk, 48, m_u);
            put_u32(&mut a_qk, 52, qkv_n);
            put_u32(&mut a_qk, 56, dim);
            put_u32(&mut a_qk, 60, ld.qkv_type as u32);
            chain.push(gpu.prepare_qmd(k_gemm, ((qkv_n + 3) / 4, grid_m, 1), (128, 1, 1), &a_qk, None));

            if ld.split_v != 0 {
                let mut a_vs = [0u8; 64];
                put_u64(&mut a_vs, 0, self.ws_dec_v_sep.va_addr);
                put_u64(&mut a_vs, 8, ld.w_v);
                put_u64(&mut a_vs, 16, self.ws_dec_xq.va_addr);
                put_u64(&mut a_vs, 24, self.ws_dec_xds.va_addr);
                put_u64(&mut a_vs, 32, 0);
                put_u64(&mut a_vs, 40, 0);
                put_u32(&mut a_vs, 48, m_u);
                put_u32(&mut a_vs, 52, kv_dim);
                put_u32(&mut a_vs, 56, dim);
                put_u32(&mut a_vs, 60, ld.v_type as u32);
                chain.push(gpu.prepare_qmd(k_gemm, ((kv_dim + 3) / 4, grid_m, 1), (128, 1, 1), &a_vs, None));
            }

            let mut a_rope = [0u8; 96];
            put_u64(&mut a_rope, 0, self.ws_dec_q_roped.va_addr);
            put_u64(&mut a_rope, 8, ld.kv_cache);
            put_u64(&mut a_rope, 16, self.ws_dec_qkv.va_addr);
            put_u64(&mut a_rope, 24, if ld.split_v != 0 { self.ws_dec_v_sep.va_addr } else { 0 });
            put_u64(&mut a_rope, 32, ld.q_norm_w);
            put_u64(&mut a_rope, 40, ld.k_norm_w);
            put_u64(&mut a_rope, 48, self.dec_freqs_va);
            put_u32(&mut a_rope, 56, m_u);
            put_u32(&mut a_rope, 60, start_pos as u32);
            put_u32(&mut a_rope, 64, n_heads);
            put_u32(&mut a_rope, 68, n_kv_heads);
            put_u32(&mut a_rope, 72, head_dim);
            put_u32(&mut a_rope, 76, head_dim);
            put_u32(&mut a_rope, 80, max_ctx_u);
            put_u32(&mut a_rope, 84, eps_bits);
            put_u32(&mut a_rope, 88, 1);
            put_u32(&mut a_rope, 92, ld.split_v as u32);
            chain.push(gpu.prepare_qmd(k_qknorm, (n_heads + 2 * n_kv_heads, m_u, 1), (256, 1, 1), &a_rope, None));

            // On the final chunk of `0 .. T_prompt - 1`, layer 27 only needs KV written — the
            // subsequent persistent megakernel step starts at `T_prompt - 1`!
            let mut a_fl = [0u8; 52];
            put_u64(&mut a_fl, 0, self.ws_dec_attn_out.va_addr);
            put_u64(&mut a_fl, 8, self.ws_dec_q_roped.va_addr);
            put_u64(&mut a_fl, 16, ld.kv_cache);
            put_u32(&mut a_fl, 24, m_u);
            put_u32(&mut a_fl, 28, start_pos as u32);
            put_u32(&mut a_fl, 32, n_heads);
            put_u32(&mut a_fl, 36, n_kv_heads);
            put_u32(&mut a_fl, 40, head_dim);
            put_u32(&mut a_fl, 44, max_ctx_u);
            put_u32(&mut a_fl, 48, scale_bits);
            chain.push(gpu.prepare_qmd(k_flash, (n_heads, m_u, 1), (128, 1, 1), &a_fl, None));

            let attn_blks = ((t_len as u32) * q_dim) / 32;
            let mut a_q8 = [0u8; 28];
            put_u64(&mut a_q8, 0, self.ws_dec_xq.va_addr);
            put_u64(&mut a_q8, 8, self.ws_dec_xds.va_addr);
            put_u64(&mut a_q8, 16, self.ws_dec_attn_out.va_addr);
            put_u32(&mut a_q8, 24, attn_blks);
            chain.push(gpu.prepare_qmd(k_quant, ((attn_blks + 3) / 4, 1, 1), (128, 1, 1), &a_q8, None));

            let mut a_wo = [0u8; 64];
            put_u64(&mut a_wo, 0, self.ws_dec_x_res2.va_addr);
            put_u64(&mut a_wo, 8, ld.w_o);
            put_u64(&mut a_wo, 16, self.ws_dec_xq.va_addr);
            put_u64(&mut a_wo, 24, self.ws_dec_xds.va_addr);
            put_u64(&mut a_wo, 32, 0);
            put_u64(&mut a_wo, 40, self.ws_dec_x_res.va_addr);
            put_u32(&mut a_wo, 48, m_u);
            put_u32(&mut a_wo, 52, dim);
            put_u32(&mut a_wo, 56, q_dim);
            put_u32(&mut a_wo, 60, ld.o_type as u32);
            chain.push(gpu.prepare_qmd(k_gemm, ((dim + 3) / 4, grid_m, 1), (128, 1, 1), &a_wo, None));

            let mut a_rms2 = [0u8; 40];
            put_u64(&mut a_rms2, 0, self.ws_dec_xq.va_addr);
            put_u64(&mut a_rms2, 8, self.ws_dec_xds.va_addr);
            put_u64(&mut a_rms2, 16, self.ws_dec_x_res2.va_addr);
            put_u64(&mut a_rms2, 24, ld.ffn_norm_w);
            put_u32(&mut a_rms2, 32, dim);
            put_u32(&mut a_rms2, 36, eps_bits);
            chain.push(gpu.prepare_qmd(k_rms, (m_u, 1, 1), (256, 1, 1), &a_rms2, None));

            let gu_n = 2 * hidden_dim;
            let mut a_gu = [0u8; 64];
            put_u64(&mut a_gu, 0, self.ws_dec_gate_up.va_addr);
            put_u64(&mut a_gu, 8, ld.w_gate_up);
            put_u64(&mut a_gu, 16, self.ws_dec_xq.va_addr);
            put_u64(&mut a_gu, 24, self.ws_dec_xds.va_addr);
            put_u64(&mut a_gu, 32, 0);
            put_u64(&mut a_gu, 40, 0);
            put_u32(&mut a_gu, 48, m_u);
            put_u32(&mut a_gu, 52, gu_n);
            put_u32(&mut a_gu, 56, dim);
            put_u32(&mut a_gu, 60, ld.gate_up_type as u32);
            chain.push(gpu.prepare_qmd(k_gemm, ((gu_n + 3) / 4, grid_m, 1), (128, 1, 1), &a_gu, None));

            let ffn_blks = ((t_len as u32) * hidden_dim) / 32;
            let mut a_sw = [0u8; 32];
            put_u64(&mut a_sw, 0, self.ws_dec_xq.va_addr);
            put_u64(&mut a_sw, 8, self.ws_dec_xds.va_addr);
            put_u64(&mut a_sw, 16, self.ws_dec_gate_up.va_addr);
            put_u32(&mut a_sw, 24, m_u);
            put_u32(&mut a_sw, 28, hidden_dim);
            chain.push(gpu.prepare_qmd(k_swiglu, ((ffn_blks + 3) / 4, 1, 1), (128, 1, 1), &a_sw, None));

            let mut a_dn = [0u8; 64];
            put_u64(&mut a_dn, 0, self.ws_dec_x_res.va_addr);
            put_u64(&mut a_dn, 8, ld.w_down);
            put_u64(&mut a_dn, 16, self.ws_dec_xq.va_addr);
            put_u64(&mut a_dn, 24, self.ws_dec_xds.va_addr);
            put_u64(&mut a_dn, 32, 0);
            put_u64(&mut a_dn, 40, self.ws_dec_x_res2.va_addr);
            put_u32(&mut a_dn, 48, m_u);
            put_u32(&mut a_dn, 52, dim);
            put_u32(&mut a_dn, 56, hidden_dim);
            put_u32(&mut a_dn, 60, ld.down_type as u32);
            chain.push(gpu.prepare_qmd(k_gemm, ((dim + 3) / 4, grid_m, 1), (128, 1, 1), &a_dn, None));
        }

        Self::execute_qmd_chain(gpu, &chain, 30)
    }

    /// Transcribe `16,000 Hz` mono `f32` PCM 100.0% on the RTX 3080!
    pub fn transcribe(
        &mut self,
        gpu: &mut NvGpu,
        module: &CubinModule,
        pcm_16k: &[f32],
        language_hint: Option<&str>,
    ) -> Result<AsrStats, String> {
        let t_total = Instant::now();
        if pcm_16k.len() < 320 {
            return Err(format!(
                "Audio clip too short ({} samples @ 16 kHz; need >= 320 samples / 20 ms)",
                pcm_16k.len()
            ));
        }
        let n_samples = pcm_16k.len().min(ASR_MAX_SAMPLES);
        let pcm_slice = &pcm_16k[..n_samples];
        let audio_sec = (n_samples as f64) / (ASR_SAMPLE_RATE as f64);

        let n_mel_frames = n_samples / 160;
        let mel_per_chunk = self.enc_n_window * 2; // 100
        let n_chunks = (n_mel_frames + mel_per_chunk - 1) / mel_per_chunk;
        let mut last_real_mel = n_mel_frames - (n_chunks - 1) * mel_per_chunk;
        if last_real_mel == 0 {
            last_real_mel = mel_per_chunk;
        }
        let t_per_chunk = aftercnn_len(mel_per_chunk); // 13
        let last_aftercnn = aftercnn_len(last_real_mel);
        let t_enc = (n_chunks - 1) * t_per_chunk + last_aftercnn;

        // ---------------------------------------------------------------------
        // Stage 1 & 2: Upload raw 16kHz PCM + GPU STFT/Mel + 3x Conv2d Subsampler + Sinusoid PE
        // ---------------------------------------------------------------------
        let t_mel0 = Instant::now();
        let mut pcm_bytes = Vec::with_capacity(n_samples * 4);
        for &s in pcm_slice {
            pcm_bytes.extend_from_slice(&s.to_bits().to_le_bytes());
        }
        let pcm_view = VramBuffer {
            va_addr: self.ws_pcm.va_addr,
            size: self.ws_pcm.size,
            paddrs: self.ws_pcm.paddrs.clone(),
        };
        gpu.upload_to_vram(&pcm_view, 0, &pcm_bytes)?;

        let k_init_max = &module.kernels["asr_init_max_int_kernel"];
        let k_mel = &module.kernels["asr_mel_stft_128_kernel"];
        let k_mel_pack = &module.kernels["asr_mel_norm_pack_kernel"];
        let k_conv2d = &module.kernels["asr_conv2d_3x3_s2_gelu_kernel"];
        let k_perm_q8 = &module.kernels["asr_conv2_permute_q8_1_kernel"];
        let k_gemm = &module.kernels["qwen_prefill_gemm"];
        let k_add_pe = &module.kernels["asr_add_pe_trim_kernel"];

        let mut sub_chain = Vec::with_capacity(16);

        let mut a_imax = [0u8; 8];
        put_u64(&mut a_imax, 0, self.ws_max_int.va_addr);
        sub_chain.push(gpu.prepare_qmd(k_init_max, (1, 1, 1), (32, 1, 1), &a_imax, None));

        let mut a_mel = [0u8; 56];
        put_u64(&mut a_mel, 0, self.ws_log_mel.va_addr);
        put_u64(&mut a_mel, 8, self.ws_max_int.va_addr);
        put_u64(&mut a_mel, 16, self.ws_pcm.va_addr);
        put_u64(&mut a_mel, 24, self.fe_window_va);
        put_u64(&mut a_mel, 32, self.fe_dft_cos_sin_va);
        put_u64(&mut a_mel, 40, self.fe_mel_fb_va);
        put_u32(&mut a_mel, 48, n_samples as u32);
        put_u32(&mut a_mel, 52, n_mel_frames as u32);
        sub_chain.push(gpu.prepare_qmd(k_mel, (n_mel_frames as u32, 1, 1), (256, 1, 1), &a_mel, None));

        let total_pack = (n_chunks * self.enc_n_mels * mel_per_chunk) as u32;
        let mut a_pk = [0u8; 44];
        put_u64(&mut a_pk, 0, self.ws_mel_chunks.va_addr);
        put_u64(&mut a_pk, 8, self.ws_log_mel.va_addr);
        put_u64(&mut a_pk, 16, self.ws_max_int.va_addr);
        put_u32(&mut a_pk, 24, self.enc_n_mels as u32);
        put_u32(&mut a_pk, 28, n_mel_frames as u32);
        put_u32(&mut a_pk, 32, n_chunks as u32);
        put_u32(&mut a_pk, 36, mel_per_chunk as u32);
        put_u32(&mut a_pk, 40, last_real_mel as u32);
        sub_chain.push(gpu.prepare_qmd(k_mel_pack, ((total_pack + 255) / 256, 1, 1), (256, 1, 1), &a_pk, None));

        let ds_h = self.enc_ds_hidden as u32;
        let h0 = self.enc_n_mels as u32; // 128
        let w0 = mel_per_chunk as u32; // 100
        let h1 = (h0 + 1) / 2; // 64
        let w1 = (w0 + 1) / 2; // 50
        let h2 = (h1 + 1) / 2; // 32
        let w2 = (w1 + 1) / 2; // 25
        let h3 = (h2 + 1) / 2; // 16
        let w3 = (w2 + 1) / 2; // 13
        let conv_out_in = ds_h * h3; // 7680
        let d_model_u = self.enc_d_model as u32;

        let mut c0 = 0usize;
        while c0 < n_chunks {
            let grp = (n_chunks - c0).min(ASR_CONV_GROUP_CHUNKS);
            let grp_u = grp as u32;
            let mel_in_va =
                self.ws_mel_chunks.va_addr + (c0 * self.enc_n_mels * mel_per_chunk * 4) as u64;

            // conv0: [grp, 1, 128, 100] -> ws_conv_a [grp, 480, 64, 50]
            let sp1 = grp_u * h1 * w1;
            let mut ac0 = [0u8; 60];
            put_u64(&mut ac0, 0, self.ws_conv_a.va_addr);
            put_u64(&mut ac0, 8, mel_in_va);
            put_u64(&mut ac0, 16, self.conv0_w_va);
            put_u64(&mut ac0, 24, self.conv0_b_va);
            put_u32(&mut ac0, 32, grp_u);
            put_u32(&mut ac0, 36, 1);
            put_u32(&mut ac0, 40, ds_h);
            put_u32(&mut ac0, 44, h0);
            put_u32(&mut ac0, 48, w0);
            put_u32(&mut ac0, 52, h1);
            put_u32(&mut ac0, 56, w1);
            sub_chain.push(gpu.prepare_qmd(
                k_conv2d,
                ((ds_h + 3) / 4, (sp1 + 3) / 4, 1),
                (128, 1, 1),
                &ac0,
                None,
            ));

            // conv1: ws_conv_a -> ws_conv_b [grp, 480, 32, 25]
            let sp2 = grp_u * h2 * w2;
            let mut ac1 = [0u8; 60];
            put_u64(&mut ac1, 0, self.ws_conv_b.va_addr);
            put_u64(&mut ac1, 8, self.ws_conv_a.va_addr);
            put_u64(&mut ac1, 16, self.conv1_w_va);
            put_u64(&mut ac1, 24, self.conv1_b_va);
            put_u32(&mut ac1, 32, grp_u);
            put_u32(&mut ac1, 36, ds_h);
            put_u32(&mut ac1, 40, ds_h);
            put_u32(&mut ac1, 44, h1);
            put_u32(&mut ac1, 48, w1);
            put_u32(&mut ac1, 52, h2);
            put_u32(&mut ac1, 56, w2);
            sub_chain.push(gpu.prepare_qmd(
                k_conv2d,
                ((ds_h + 3) / 4, (sp2 + 3) / 4, 1),
                (128, 1, 1),
                &ac1,
                None,
            ));

            // conv2: ws_conv_b -> ws_conv_a [grp, 480, 16, 13]
            let sp3 = grp_u * h3 * w3;
            let mut ac2 = [0u8; 60];
            put_u64(&mut ac2, 0, self.ws_conv_a.va_addr);
            put_u64(&mut ac2, 8, self.ws_conv_b.va_addr);
            put_u64(&mut ac2, 16, self.conv2_w_va);
            put_u64(&mut ac2, 24, self.conv2_b_va);
            put_u32(&mut ac2, 32, grp_u);
            put_u32(&mut ac2, 36, ds_h);
            put_u32(&mut ac2, 40, ds_h);
            put_u32(&mut ac2, 44, h2);
            put_u32(&mut ac2, 48, w2);
            put_u32(&mut ac2, 52, h3);
            put_u32(&mut ac2, 56, w3);
            sub_chain.push(gpu.prepare_qmd(
                k_conv2d,
                ((ds_h + 3) / 4, (sp3 + 3) / 4, 1),
                (128, 1, 1),
                &ac2,
                None,
            ));

            // Permute + quantize to Q8_1 (`[grp * 13, 7680]`)
            let m_grp = grp_u * w3;
            let perm_blks = m_grp * (conv_out_in / 32);
            let mut apq = [0u8; 40];
            put_u64(&mut apq, 0, self.ws_dec_xq.va_addr);
            put_u64(&mut apq, 8, self.ws_dec_xds.va_addr);
            put_u64(&mut apq, 16, self.ws_conv_a.va_addr);
            put_u32(&mut apq, 24, grp_u);
            put_u32(&mut apq, 28, ds_h);
            put_u32(&mut apq, 32, h3);
            put_u32(&mut apq, 36, w3);
            sub_chain.push(gpu.prepare_qmd(
                k_perm_q8,
                ((perm_blks + 3) / 4, 1, 1),
                (128, 1, 1),
                &apq,
                None,
            ));

            // Linear `enc.conv_out` (`[grp * 13, 7680] -> ws_conv_out_pad[c0 * 13 .., d_model]`)
            let dst_co_va =
                self.ws_conv_out_pad.va_addr + (c0 * t_per_chunk * self.enc_d_model * 4) as u64;
            let mut aco = [0u8; 64];
            put_u64(&mut aco, 0, dst_co_va);
            put_u64(&mut aco, 8, self.conv_out_w_va);
            put_u64(&mut aco, 16, self.ws_dec_xq.va_addr);
            put_u64(&mut aco, 24, self.ws_dec_xds.va_addr);
            put_u64(&mut aco, 32, 0);
            put_u64(&mut aco, 40, 0);
            put_u32(&mut aco, 48, m_grp);
            put_u32(&mut aco, 52, d_model_u);
            put_u32(&mut aco, 56, conv_out_in);
            put_u32(&mut aco, 60, self.conv_out_type as u32);
            sub_chain.push(gpu.prepare_qmd(
                k_gemm,
                ((d_model_u + 3) / 4, (m_grp + 7) / 8, 1),
                (128, 1, 1),
                &aco,
                None,
            ));

            c0 += grp;
        }

        // Add Sinusoidal PE & trim ragged tail to `ws_enc_x [T_enc, d_model]`
        let mut ape = [0u8; 36];
        put_u64(&mut ape, 0, self.ws_enc_x.va_addr);
        put_u64(&mut ape, 8, self.ws_conv_out_pad.va_addr);
        put_u64(&mut ape, 16, self.pos_emb_va);
        put_u32(&mut ape, 24, t_enc as u32);
        put_u32(&mut ape, 28, t_per_chunk as u32);
        put_u32(&mut ape, 32, d_model_u);
        sub_chain.push(gpu.prepare_qmd(k_add_pe, (t_enc as u32, 1, 1), (256, 1, 1), &ape, None));

        Self::execute_qmd_chain(gpu, &sub_chain, 30)?;
        let mel_subsample_ms = t_mel0.elapsed().as_secs_f64() * 1000.0;

        // ---------------------------------------------------------------------
        // Stage 3: 24-Layer Bidirectional Flash-Attention AuT Encoder + Projector (1 QMD Chain!)
        // ---------------------------------------------------------------------
        let t_enc0 = Instant::now();
        let k_ln_q8 = &module.kernels["asr_layernorm_q8_1_kernel"];
        let k_bi_attn = &module.kernels["asr_bidir_flash_attn_f32_kernel"];
        let k_quant = &module.kernels["qwen_prefill_quant_q8_1"];
        let k_gelu_q8 = &module.kernels["asr_gelu_erf_q8_1_kernel"];

        let t_enc_u = t_enc as u32;
        let grid_t = (t_enc_u + 7) / 8;
        let ffn_u = self.enc_ffn_dim as u32;
        let out_dim_u = self.enc_out_dim as u32;
        let ln_eps_bits = (1.0e-5f32).to_bits();
        let attn_scale_bits = (0.125f32).to_bits(); // 1 / sqrt(64)
        let d_blks = (t_enc_u * d_model_u) / 32;
        let ffn_blks = (t_enc_u * ffn_u) / 32;

        let mut enc_chain = Vec::with_capacity(self.enc_n_layers * 10 + 8);
        for blk in &self.enc_blocks {
            // 1. Pre-LN -> (xq, xds)
            let mut aln1 = [0u8; 56];
            put_u64(&mut aln1, 0, self.ws_dec_xq.va_addr);
            put_u64(&mut aln1, 8, self.ws_dec_xds.va_addr);
            put_u64(&mut aln1, 16, 0);
            put_u64(&mut aln1, 24, self.ws_enc_x.va_addr);
            put_u64(&mut aln1, 32, blk.norm_attn_w);
            put_u64(&mut aln1, 40, blk.norm_attn_b);
            put_u32(&mut aln1, 48, d_model_u);
            put_u32(&mut aln1, 52, ln_eps_bits);
            enc_chain.push(gpu.prepare_qmd(k_ln_q8, (t_enc_u, 1, 1), (256, 1, 1), &aln1, None));

            // 2. Q, K, V projections
            for &(dst_va, w_va, b_va, wt) in &[
                (self.ws_enc_q.va_addr, blk.attn_q_w, blk.attn_q_b, blk.attn_q_type),
                (self.ws_enc_k.va_addr, blk.attn_k_w, blk.attn_k_b, blk.attn_k_type),
                (self.ws_enc_v.va_addr, blk.attn_v_w, blk.attn_v_b, blk.attn_v_type),
            ] {
                let mut a_qkv = [0u8; 64];
                put_u64(&mut a_qkv, 0, dst_va);
                put_u64(&mut a_qkv, 8, w_va);
                put_u64(&mut a_qkv, 16, self.ws_dec_xq.va_addr);
                put_u64(&mut a_qkv, 24, self.ws_dec_xds.va_addr);
                put_u64(&mut a_qkv, 32, b_va);
                put_u64(&mut a_qkv, 40, 0);
                put_u32(&mut a_qkv, 48, t_enc_u);
                put_u32(&mut a_qkv, 52, d_model_u);
                put_u32(&mut a_qkv, 56, d_model_u);
                put_u32(&mut a_qkv, 60, wt as u32);
                enc_chain.push(gpu.prepare_qmd(
                    k_gemm,
                    ((d_model_u + 3) / 4, grid_t, 1),
                    (128, 1, 1),
                    &a_qkv,
                    None,
                ));
            }

            // 3. Bidirectional Online-Softmax Flash Attention (`ws_enc_q, k, v` -> `ws_enc_x2`)
            let mut aba = [0u8; 44];
            put_u64(&mut aba, 0, self.ws_enc_x2.va_addr);
            put_u64(&mut aba, 8, self.ws_enc_q.va_addr);
            put_u64(&mut aba, 16, self.ws_enc_k.va_addr);
            put_u64(&mut aba, 24, self.ws_enc_v.va_addr);
            put_u32(&mut aba, 32, t_enc_u);
            put_u32(&mut aba, 36, self.enc_n_heads as u32);
            put_u32(&mut aba, 40, attn_scale_bits);
            enc_chain.push(gpu.prepare_qmd(
                k_bi_attn,
                (self.enc_n_heads as u32, t_enc_u, 1),
                (128, 1, 1),
                &aba,
                None,
            ));

            // 4. Quantize `ws_enc_x2` -> `(xq, xds)` + `attn.out` (+ residual `ws_enc_x` -> `ws_enc_x2`)
            let mut aq1 = [0u8; 28];
            put_u64(&mut aq1, 0, self.ws_dec_xq.va_addr);
            put_u64(&mut aq1, 8, self.ws_dec_xds.va_addr);
            put_u64(&mut aq1, 16, self.ws_enc_x2.va_addr);
            put_u32(&mut aq1, 24, d_blks);
            enc_chain.push(gpu.prepare_qmd(k_quant, ((d_blks + 3) / 4, 1, 1), (128, 1, 1), &aq1, None));

            let mut a_ao = [0u8; 64];
            put_u64(&mut a_ao, 0, self.ws_enc_x2.va_addr);
            put_u64(&mut a_ao, 8, blk.attn_out_w);
            put_u64(&mut a_ao, 16, self.ws_dec_xq.va_addr);
            put_u64(&mut a_ao, 24, self.ws_dec_xds.va_addr);
            put_u64(&mut a_ao, 32, blk.attn_out_b);
            put_u64(&mut a_ao, 40, self.ws_enc_x.va_addr);
            put_u32(&mut a_ao, 48, t_enc_u);
            put_u32(&mut a_ao, 52, d_model_u);
            put_u32(&mut a_ao, 56, d_model_u);
            put_u32(&mut a_ao, 60, blk.attn_out_type as u32);
            enc_chain.push(gpu.prepare_qmd(
                k_gemm,
                ((d_model_u + 3) / 4, grid_t, 1),
                (128, 1, 1),
                &a_ao,
                None,
            ));

            // 5. FFN Pre-LN (`ws_enc_x2` -> `xq, xds`) -> `fc1` -> `GELU_erf + Q8_1` -> `fc2` (+ residual `ws_enc_x2` -> `ws_enc_x`)
            let mut aln2 = [0u8; 56];
            put_u64(&mut aln2, 0, self.ws_dec_xq.va_addr);
            put_u64(&mut aln2, 8, self.ws_dec_xds.va_addr);
            put_u64(&mut aln2, 16, 0);
            put_u64(&mut aln2, 24, self.ws_enc_x2.va_addr);
            put_u64(&mut aln2, 32, blk.norm_ffn_w);
            put_u64(&mut aln2, 40, blk.norm_ffn_b);
            put_u32(&mut aln2, 48, d_model_u);
            put_u32(&mut aln2, 52, ln_eps_bits);
            enc_chain.push(gpu.prepare_qmd(k_ln_q8, (t_enc_u, 1, 1), (256, 1, 1), &aln2, None));

            let mut af1 = [0u8; 64];
            put_u64(&mut af1, 0, self.ws_enc_ffn.va_addr);
            put_u64(&mut af1, 8, blk.fc1_w);
            put_u64(&mut af1, 16, self.ws_dec_xq.va_addr);
            put_u64(&mut af1, 24, self.ws_dec_xds.va_addr);
            put_u64(&mut af1, 32, blk.fc1_b);
            put_u64(&mut af1, 40, 0);
            put_u32(&mut af1, 48, t_enc_u);
            put_u32(&mut af1, 52, ffn_u);
            put_u32(&mut af1, 56, d_model_u);
            put_u32(&mut af1, 60, blk.fc1_type as u32);
            enc_chain.push(gpu.prepare_qmd(
                k_gemm,
                ((ffn_u + 3) / 4, grid_t, 1),
                (128, 1, 1),
                &af1,
                None,
            ));

            let mut agq = [0u8; 28];
            put_u64(&mut agq, 0, self.ws_dec_xq.va_addr);
            put_u64(&mut agq, 8, self.ws_dec_xds.va_addr);
            put_u64(&mut agq, 16, self.ws_enc_ffn.va_addr);
            put_u32(&mut agq, 24, ffn_blks);
            enc_chain.push(gpu.prepare_qmd(k_gelu_q8, ((ffn_blks + 3) / 4, 1, 1), (128, 1, 1), &agq, None));

            let mut af2 = [0u8; 64];
            put_u64(&mut af2, 0, self.ws_enc_x.va_addr);
            put_u64(&mut af2, 8, blk.fc2_w);
            put_u64(&mut af2, 16, self.ws_dec_xq.va_addr);
            put_u64(&mut af2, 24, self.ws_dec_xds.va_addr);
            put_u64(&mut af2, 32, blk.fc2_b);
            put_u64(&mut af2, 40, self.ws_enc_x2.va_addr);
            put_u32(&mut af2, 48, t_enc_u);
            put_u32(&mut af2, 52, d_model_u);
            put_u32(&mut af2, 56, ffn_u);
            put_u32(&mut af2, 60, blk.fc2_type as u32);
            enc_chain.push(gpu.prepare_qmd(
                k_gemm,
                ((d_model_u + 3) / 4, grid_t, 1),
                (128, 1, 1),
                &af2,
                None,
            ));
        }

        // Post-Head: `ln_post` -> `proj1` -> `GELU_erf + Q8_1` -> `proj2` -> `ws_enc_proj_out [T_enc, dec_hidden]`
        let mut alnp = [0u8; 56];
        put_u64(&mut alnp, 0, self.ws_dec_xq.va_addr);
        put_u64(&mut alnp, 8, self.ws_dec_xds.va_addr);
        put_u64(&mut alnp, 16, 0);
        put_u64(&mut alnp, 24, self.ws_enc_x.va_addr);
        put_u64(&mut alnp, 32, self.ln_post_w_va);
        put_u64(&mut alnp, 40, self.ln_post_b_va);
        put_u32(&mut alnp, 48, d_model_u);
        put_u32(&mut alnp, 52, ln_eps_bits);
        enc_chain.push(gpu.prepare_qmd(k_ln_q8, (t_enc_u, 1, 1), (256, 1, 1), &alnp, None));

        let mut ap1 = [0u8; 64];
        put_u64(&mut ap1, 0, self.ws_enc_x2.va_addr);
        put_u64(&mut ap1, 8, self.proj1_w_va);
        put_u64(&mut ap1, 16, self.ws_dec_xq.va_addr);
        put_u64(&mut ap1, 24, self.ws_dec_xds.va_addr);
        put_u64(&mut ap1, 32, self.proj1_b_va);
        put_u64(&mut ap1, 40, 0);
        put_u32(&mut ap1, 48, t_enc_u);
        put_u32(&mut ap1, 52, d_model_u);
        put_u32(&mut ap1, 56, d_model_u);
        put_u32(&mut ap1, 60, self.proj1_type as u32);
        enc_chain.push(gpu.prepare_qmd(
            k_gemm,
            ((d_model_u + 3) / 4, grid_t, 1),
            (128, 1, 1),
            &ap1,
            None,
        ));

        let mut apg = [0u8; 28];
        put_u64(&mut apg, 0, self.ws_dec_xq.va_addr);
        put_u64(&mut apg, 8, self.ws_dec_xds.va_addr);
        put_u64(&mut apg, 16, self.ws_enc_x2.va_addr);
        put_u32(&mut apg, 24, d_blks);
        enc_chain.push(gpu.prepare_qmd(k_gelu_q8, ((d_blks + 3) / 4, 1, 1), (128, 1, 1), &apg, None));

        let mut ap2 = [0u8; 64];
        put_u64(&mut ap2, 0, self.ws_enc_proj_out.va_addr);
        put_u64(&mut ap2, 8, self.proj2_w_va);
        put_u64(&mut ap2, 16, self.ws_dec_xq.va_addr);
        put_u64(&mut ap2, 24, self.ws_dec_xds.va_addr);
        put_u64(&mut ap2, 32, self.proj2_b_va);
        put_u64(&mut ap2, 40, 0);
        put_u32(&mut ap2, 48, t_enc_u);
        put_u32(&mut ap2, 52, out_dim_u);
        put_u32(&mut ap2, 56, d_model_u);
        put_u32(&mut ap2, 60, self.proj2_type as u32);
        enc_chain.push(gpu.prepare_qmd(
            k_gemm,
            ((out_dim_u + 3) / 4, grid_t, 1),
            (128, 1, 1),
            &ap2,
            None,
        ));

        Self::execute_qmd_chain(gpu, &enc_chain, 30)?;
        let encoder_ms = t_enc0.elapsed().as_secs_f64() * 1000.0;

        // ---------------------------------------------------------------------
        // Stage 4: Build ChatML + Audio Prompt, Prefill `[0 .. T_prompt - 1)`, & Launch Persistent Megakernel
        // ---------------------------------------------------------------------
        let t_pf0 = Instant::now();
        let resolved_lang = language_hint.and_then(normalize_lang_hint);

        let mut prompt_ids = Vec::with_capacity(24 + t_enc);
        prompt_ids.extend_from_slice(&[
            self.im_start_id,
            self.role_system_id,
            self.newline_id,
            self.im_end_id,
            self.newline_id,
            self.im_start_id,
            self.role_user_id,
            self.newline_id,
            self.audio_start_id,
        ]);
        let prefix_len = prompt_ids.len();
        prompt_ids.resize(prefix_len + t_enc, self.audio_pad_id);
        prompt_ids.extend_from_slice(&[
            self.audio_end_id,
            self.im_end_id,
            self.newline_id,
            self.im_start_id,
            self.role_assistant_id,
            self.newline_id,
        ]);
        if let Some((_, pub_name)) = resolved_lang {
            prompt_ids.extend(self.tok.encode(&format!("language {}", pub_name)));
            prompt_ids.push(self.asr_text_id);
        }

        let t_prompt = prompt_ids.len();
        let t_pf = t_prompt - 1;
        let last_prompt_tok = prompt_ids[t_pf];
        let max_new = (ASR_DEC_MAX_CTX.saturating_sub(t_prompt + 4)).min(384);

        {
            let dst = self.host_prompt_buf.as_mut_slice();
            for (i, &id) in prompt_ids[..t_pf].iter().enumerate() {
                put_u32(dst, i * 4, id as u32);
            }
        }
        fence(Ordering::SeqCst);

        // Assemble `[0 .. T_prompt - 1)` embeddings directly in VRAM
        let k_asm_dec = &module.kernels["asr_assemble_decoder_embed_kernel"];
        let mut a_asm = [0u8; 52];
        put_u64(&mut a_asm, 0, self.ws_dec_prefill_emb.va_addr);
        put_u64(&mut a_asm, 8, self.dec_emb_va);
        put_u64(&mut a_asm, 16, self.host_prompt_buf.va_addr);
        put_u64(&mut a_asm, 24, self.ws_enc_proj_out.va_addr);
        put_u32(&mut a_asm, 32, t_pf as u32);
        put_u32(&mut a_asm, 36, prefix_len as u32);
        put_u32(&mut a_asm, 40, t_enc as u32);
        put_u32(&mut a_asm, 44, self.dec_hidden as u32);
        put_u32(&mut a_asm, 48, self.dec_emb_type as u32);
        let asm_qmd = [gpu.prepare_qmd(k_asm_dec, (t_pf as u32, 1, 1), (256, 1, 1), &a_asm, None)];
        Self::execute_qmd_chain(gpu, &asm_qmd, 15)?;

        let mut pos = 0usize;
        while pos < t_pf {
            let take = (t_pf - pos).min(ASR_PREFILL_CHUNK);
            self.run_dec_prefill_chunk(gpu, module, pos, take)?;
            pos += take;
        }
        let decoder_prefill_ms = t_pf0.elapsed().as_secs_f64() * 1000.0;

        // Launch ONE persistent 68-SM `qwen_persistent_decode_megakernel` to emit the full transcript!
        let t_dec0 = Instant::now();
        let ctrl_ptr = self.dec_ctrl_buf.as_ptr() as *mut QwenDecodeCtrl;
        unsafe {
            (*ctrl_ptr).start_token = last_prompt_tok;
            (*ctrl_ptr).start_pos = t_pf as i32;
            (*ctrl_ptr).max_steps = max_new as i32;
            (*ctrl_ptr).temperature = 0.0;
            (*ctrl_ptr).rng_seed = 42;
            (*ctrl_ptr).cancel_flag = 0;
            (*ctrl_ptr).error_code = 0;
            (*ctrl_ptr).steps_completed = 0;
            (*ctrl_ptr).stop_reason = 0;
            (*ctrl_ptr).export_logits = 0;
        }
        fence(Ordering::SeqCst);

        let k_mk = &module.kernels["qwen_persistent_decode_megakernel"];
        let mut mk_args = [0u8; 16];
        put_u64(&mut mk_args, 0, self.dec_model_desc_gpu.va_addr);
        put_u64(&mut mk_args, 8, self.dec_ctrl_buf.va_addr);
        let mk_qmd = [gpu.prepare_qmd(k_mk, (gpu.active_sms, 1, 1), (512, 1, 1), &mk_args, None)];
        Self::execute_qmd_chain(gpu, &mk_qmd, 30)?;
        let decoder_decode_ms = t_dec0.elapsed().as_secs_f64() * 1000.0;

        let (err_code, steps_done) = unsafe {
            (
                std::ptr::read_volatile(&(*ctrl_ptr).error_code),
                std::ptr::read_volatile(&(*ctrl_ptr).steps_completed) as usize,
            )
        };
        if err_code != 0 {
            let _ = gpu.zero_vram(&self.ws_dec_bar_state);
            return Err(format!("ASR decoder megakernel watchdog tripped (err={})", err_code));
        }

        let mut gen_ids = Vec::with_capacity(steps_done);
        for s in 0..steps_done {
            let tid = unsafe { std::ptr::read_volatile(&(*ctrl_ptr).tokens_out[s]) };
            if tid == self.im_end_id || tid == self.eos_id {
                break;
            }
            gen_ids.push(tid);
        }

        // Decode UTF-8 & parse `"language <Name><asr_text>..."` envelope
        let mut utf8 = Utf8StreamDecoder::new();
        let mut raw_text = String::new();
        for &tid in &gen_ids {
            raw_text.push_str(&utf8.push(self.tok.token_bytes(tid)));
        }
        raw_text.push_str(&utf8.flush());

        let mut detected_lang = resolved_lang
            .map(|(code, _)| code.to_string())
            .unwrap_or_else(|| "auto".to_string());
        let transcript = if let Some(sep) = raw_text.find("<asr_text>") {
            if resolved_lang.is_none() {
                if let Some(ls) = raw_text[..sep].find("language ") {
                    let pub_name = raw_text[ls + 9..sep].trim();
                    for &(code, pname) in LANG_NAME_MAP {
                        if pname.eq_ignore_ascii_case(pub_name) {
                            detected_lang = code.to_string();
                            break;
                        }
                    }
                }
            }
            raw_text[sep + "<asr_text>".len()..].trim().to_string()
        } else {
            raw_text.trim().to_string()
        };

        let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
        let rtf = if audio_sec > 0.0 {
            (total_ms / 1000.0) / audio_sec
        } else {
            0.0
        };
        let speed_x_realtime = if rtf > 0.0 { 1.0 / rtf } else { 0.0 };

        Ok(AsrStats {
            asr_model: self.model_name.clone(),
            text: transcript,
            detected_language: detected_lang,
            audio_sec,
            n_samples,
            mel_frames: n_mel_frames,
            audio_tokens: t_enc,
            prompt_tokens: t_prompt,
            completion_tokens: gen_ids.len(),
            mel_subsample_ms,
            encoder_ms,
            decoder_prefill_ms,
            decoder_decode_ms,
            total_ms,
            rtf,
            speed_x_realtime,
        })
    }
}
