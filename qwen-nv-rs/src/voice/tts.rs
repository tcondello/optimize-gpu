//! 100%-on-RTX-3080 (`sm_86`) `Qwen3-TTS-12Hz-1.7B` (`Talker` + `CodePredictor` + `12Hz Code2Wav`) Engine.
//! Zero neural layers execute on the host CPU/M3 Pro: prompt projection, 28L Talker, top-k/rep-pen
//! sampling, 5L x 15-step CodePredictor, 16-codebook embedding composition, Split-RVQ, 8L Sliding-Window
//! Transformer, 2x ConvNeXt Upsampler, and 4-block SnakeBeta DAC Vocoder ALL execute in RTX 3080 GDDR6X!

#![allow(dead_code)]

use std::sync::atomic::{fence, AtomicBool, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use super::audio::{TTS_HOP_SAMPLES, TTS_SAMPLE_RATE};
pub use super::code2wav::TTS_MAX_CODEC_CHUNK_FRAMES;
use super::code2wav::{ByteSlab, Code2WavEngine};
use crate::gguf::{
    decode_1d_f32, extract_quant_weight_bytes, precompute_freqs_cis_f32, resolve_tts_model_paths,
    BpeTokenizer, GgufFile,
};
use crate::hw::{put_u32, put_u64, CubinModule, HostMappedBuffer, NvGpu, VramBuffer};
use crate::models::qwen_abi::{QwenDecodeCtrl, QwenLayerDesc, QwenModelDesc};

const TALKER_PREFILL_CHUNK: usize = 64;

// -----------------------------------------------------------------------------
// Exact Host/Device ABI Structs Matching Section 5 of `qwen_sm86_megakernel.cu`
// -----------------------------------------------------------------------------

#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct TtsCodePredDesc {
    pub layers_ptr: u64,
    pub mtp_proj_w: u64,
    pub mtp_proj_b: u64,
    pub output_norm_w: u64,
    pub lm_head_w: [u64; 16],
    pub cp_codec_emb_w: [u64; 16],
    pub talker_codec_emb_w: u64,
    pub freqs_cis: u64,
    pub talker_hidden_last: u64,
    pub tts_pad_embed: u64,
    pub talker_ws_ext_embed: u64,

    pub ws_in2048: u64,
    pub ws_x_res: u64,
    pub ws_x_res2: u64,
    pub ws_xq: u64,
    pub ws_xds: u64,
    pub ws_qkv: u64,
    pub ws_q_roped: u64,
    pub ws_gate_up: u64,
    pub ws_logits: u64,
    pub ws_codes_frame: u64,
    pub ws_bar_state: u64,

    pub num_layers: i32,
    pub talker_dim: i32,
    pub dim: i32,
    pub hidden_dim: i32,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub vocab_size: i32,
    pub max_ctx: i32,
    pub mtp_proj_type: i32,
    pub talker_codec_emb_type: i32,
    pub lm_head_type: i32,
    pub cp_codec_emb_type: i32,
    pub norm_eps: f32,
    pub inv_sqrt_head_dim: f32,
    pub _pad_cp: i32,
}

#[repr(C, align(64))]
pub struct TtsFrameCtrl {
    pub seed: u64,
    pub overlay_ptr: u64,
    pub frame_idx: i32,
    pub c0: i32,
    pub is_eos: i32,
    pub error_code: i32,
    pub frames_completed: i32,
    pub talker_temp: f32,
    pub talker_top_k: i32,
    pub talker_top_p: f32,
    pub rep_penalty: f32,
    pub sub_temp: f32,
    pub sub_top_k: i32,
    pub sub_top_p: f32,
    pub codec_eos_id: i32,
    pub _pad_fc: [i32; 3],
    pub codes_out: [i32; 2048 * 16],
}

#[derive(Clone, Debug)]
pub struct TtsSynthParams {
    pub text: String,
    pub speaker: String,
    pub language: String,
    pub instruct: String,
    pub max_frames: usize,
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub rep_penalty: f32,
    pub sub_temperature: f32,
    pub sub_top_k: i32,
    pub sub_top_p: f32,
    pub seed: Option<u64>,
    pub streaming_chunks: bool,
}

impl Default for TtsSynthParams {
    fn default() -> Self {
        Self {
            text: String::new(),
            speaker: "ryan".to_string(),
            language: "english".to_string(),
            instruct: String::new(),
            max_frames: 1024,
            temperature: 0.9,
            top_k: 50,
            top_p: 1.0,
            rep_penalty: 1.05,
            sub_temperature: 0.9,
            sub_top_k: 50,
            sub_top_p: 1.0,
            seed: None,
            streaming_chunks: true,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TtsStats {
    pub talker_model: String,
    pub speaker: String,
    pub language: String,
    pub prompt_ctx_rows: usize,
    pub frames: usize,
    pub n_samples: usize,
    pub audio_sec: f64,
    pub prefill_ms: f64,
    pub first_codes_ms: f64,
    pub ttfa_ms: f64,
    pub ar_decode_ms: f64,
    pub ar_ms_per_frame: f64,
    pub codec_ms: f64,
    pub total_ms: f64,
    pub rtf: f64,
    pub speed_x_realtime: f64,
}

impl TtsStats {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"talker_model\":\"{}\",\"speaker\":\"{}\",\"language\":\"{}\",\"prompt_ctx_rows\":{},\
              \"frames\":{},\"n_samples\":{},\"audio_sec\":{:.3},\"prefill_ms\":{:.3},\
              \"first_codes_ms\":{:.3},\"ttfa_ms\":{:.3},\"ar_decode_ms\":{:.3},\
              \"ar_ms_per_frame\":{:.3},\"codec_ms\":{:.3},\"total_ms\":{:.3},\
              \"rtf\":{:.4},\"speed_x_realtime\":{:.2}}}",
            self.talker_model,
            self.speaker,
            self.language,
            self.prompt_ctx_rows,
            self.frames,
            self.n_samples,
            self.audio_sec,
            self.prefill_ms,
            self.first_codes_ms,
            self.ttfa_ms,
            self.ar_decode_ms,
            self.ar_ms_per_frame,
            self.codec_ms,
            self.total_ms,
            self.rtf,
            self.speed_x_realtime
        )
    }
}

#[derive(Clone, Debug)]
struct SpeakerEntry {
    name: String,
    id: i32,
    dialect: String,
}

#[derive(Clone, Debug)]
struct LanguageEntry {
    name: String,
    id: i32,
}

pub struct QwenTtsEngine {
    talker_ref: String,
    model_name: String,
    tok: BpeTokenizer,
    _owned_vram: Vec<VramBuffer>,

    // Metadata
    speakers: Vec<SpeakerEntry>,
    languages: Vec<LanguageEntry>,
    codec_pad_id: i32,
    codec_bos_id: i32,
    codec_eos_id: i32,
    codec_think_id: i32,
    codec_nothink_id: i32,
    codec_think_bos_id: i32,
    codec_think_eos_id: i32,
    tts_pad_id: i32,
    tts_bos_id: i32,
    tts_eos_id: i32,

    // Talker (28L, dim=2048 on 1.7B / dim=1024 on 0.6B; text_emb_dim=2048 on both!)
    talker_dim: usize,
    talker_text_emb_dim: usize,
    talker_fc1_out_dim: usize,
    talker_hidden_dim: usize,
    talker_n_heads: usize,
    talker_n_kv_heads: usize,
    talker_head_dim: usize,
    talker_vocab: usize,
    talker_max_ctx: usize,
    talker_eps: f32,
    talker_layers: Vec<QwenLayerDesc>,
    talker_model_desc_gpu: VramBuffer,
    talker_freqs_va: u64,
    talker_text_emb_va: u64,
    talker_text_emb_type: i32,
    talker_fc1_w: u64,
    talker_fc1_b: u64,
    talker_fc1_type: i32,
    talker_fc2_w: u64,
    talker_fc2_b: u64,
    talker_fc2_type: i32,
    talker_codec_emb_va: u64,
    talker_codec_emb_type: i32,
    talker_out_norm_va: u64,
    talker_codec_head_va: u64,
    talker_codec_head_type: i32,

    // Talker workspaces
    tk_prefill_emb: VramBuffer,
    tk_proj_buf: VramBuffer,
    tk_tts_pad_emb: VramBuffer,
    tk_ext_embed: VramBuffer,
    tk_hidden_last: VramBuffer,
    tk_logits_out: VramBuffer,
    tk_ws_x_res: VramBuffer,
    tk_ws_x_res2: VramBuffer,
    tk_ws_xq: VramBuffer,
    tk_ws_xds: VramBuffer,
    tk_ws_qkv: VramBuffer,
    tk_ws_v_sep: VramBuffer,
    tk_ws_q_roped: VramBuffer,
    tk_ws_attn_out: VramBuffer,
    tk_ws_gate_up: VramBuffer,
    tk_ws_bar_state: VramBuffer,
    tk_history_seen: VramBuffer,

    // CodePredictor (5L, dim=1024)
    cp_desc_gpu: VramBuffer,
    cp_ws_codes_frame: VramBuffer,
    cp_ws_bar_state: VramBuffer,

    // Host-mapped control & audio output buffers (only 3 MAP_SYSMEM_FD buffers total!)
    tk_decode_ctrl: HostMappedBuffer,
    frame_ctrl: HostMappedBuffer,
    host_io_buf: HostMappedBuffer, // [0..16KB]=ids/codes staging, [16KB..]=61,440 f32 audio samples!

    c2w: Code2WavEngine,
    pub total_vram_mb: f64,
}

impl QwenTtsEngine {
    #[inline]
    pub fn talker_ref(&self) -> &str {
        &self.talker_ref
    }

    #[inline]
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    pub fn load(
        gpu: &mut NvGpu,
        talker_ref: &str,
        codec_ref: Option<&str>,
    ) -> Result<Self, String> {
        assert_eq!(std::mem::size_of::<TtsCodePredDesc>(), 480);

        let t0 = Instant::now();
        let (talker_path, codec_path) = resolve_tts_model_paths(talker_ref, codec_ref)?;
        let gf_tk = GgufFile::open(&talker_path)?;
        let gf_cd = GgufFile::open(&codec_path)?;
        let tok = gf_tk.build_tokenizer()?;

        let get_u = |gf: &GgufFile, k: &str| gf.req_usize(k);
        let get_f = |gf: &GgufFile, k: &str, def: f32| gf.get_f32_or(k, def);

        let model_name = gf_tk.get_str("general.name").unwrap_or("Qwen3-TTS-12Hz-1.7B").to_string();

        let talker_dim = get_u(&gf_tk, "qwen3-tts.talker.embedding_length")?;
        let talker_hidden_dim = get_u(&gf_tk, "qwen3-tts.talker.feed_forward_length")?;
        let talker_num_layers = get_u(&gf_tk, "qwen3-tts.talker.block_count")?;
        let talker_n_heads = get_u(&gf_tk, "qwen3-tts.talker.attention.head_count")?;
        let talker_n_kv_heads = get_u(&gf_tk, "qwen3-tts.talker.attention.head_count_kv")?;
        let talker_head_dim = get_u(&gf_tk, "qwen3-tts.talker.attention.key_length")?;
        let talker_vocab = get_u(&gf_tk, "qwen3-tts.talker.vocab_size")?;
        let talker_rope_theta = get_f(&gf_tk, "qwen3-tts.talker.rope.freq_base", 1_000_000.0);
        let talker_eps = get_f(&gf_tk, "qwen3-tts.talker.attention.layer_norm_rms_epsilon", 1e-6);
        let talker_max_ctx = 2048usize;

        let cp_dim = get_u(&gf_tk, "qwen3-tts.code_pred.embedding_length")?;
        let cp_hidden_dim = get_u(&gf_tk, "qwen3-tts.code_pred.feed_forward_length")?;
        let cp_num_layers = get_u(&gf_tk, "qwen3-tts.code_pred.block_count")?;
        let cp_n_heads = get_u(&gf_tk, "qwen3-tts.code_pred.attention.head_count")?;
        let cp_n_kv_heads = get_u(&gf_tk, "qwen3-tts.code_pred.attention.head_count_kv")?;
        let cp_head_dim = get_u(&gf_tk, "qwen3-tts.code_pred.attention.key_length")?;
        let cp_vocab = get_u(&gf_tk, "qwen3-tts.code_pred.vocab_size")?;
        let cp_rope_theta = get_f(&gf_tk, "qwen3-tts.code_pred.rope.freq_base", 1_000_000.0);
        let cp_eps = get_f(&gf_tk, "qwen3-tts.code_pred.attention.layer_norm_rms_epsilon", 1e-6);

        let codec_pad_id = get_u(&gf_tk, "qwen3-tts.codec.pad_id")? as i32;
        let codec_bos_id = get_u(&gf_tk, "qwen3-tts.codec.bos_id")? as i32;
        let codec_eos_id = get_u(&gf_tk, "qwen3-tts.codec.eos_id")? as i32;
        let codec_think_id = get_u(&gf_tk, "qwen3-tts.codec.think_id")? as i32;
        let codec_nothink_id = get_u(&gf_tk, "qwen3-tts.codec.nothink_id")? as i32;
        let codec_think_bos_id = get_u(&gf_tk, "qwen3-tts.codec.think_bos_id")? as i32;
        let codec_think_eos_id = get_u(&gf_tk, "qwen3-tts.codec.think_eos_id")? as i32;

        let tts_pad_id = get_u(&gf_tk, "qwen3-tts.text.tts_pad_id")? as i32;
        let tts_bos_id = get_u(&gf_tk, "qwen3-tts.text.tts_bos_id")? as i32;
        let tts_eos_id = get_u(&gf_tk, "qwen3-tts.text.tts_eos_id")? as i32;

        let mut languages = Vec::new();
        if let (Some(names), Some(ids)) = (
            gf_tk.get_str_arr("qwen3-tts.codec.language_names"),
            gf_tk.get_i32_arr("qwen3-tts.codec.language_ids"),
        ) {
            for (n, &id) in names.iter().zip(ids.iter()) {
                languages.push(LanguageEntry {
                    name: n.to_ascii_lowercase(),
                    id,
                });
            }
        }

        let mut speakers = Vec::new();
        if let (Some(names), Some(ids), Some(dialects)) = (
            gf_tk.get_str_arr("qwen3-tts.codec.speaker_names"),
            gf_tk.get_i32_arr("qwen3-tts.codec.speaker_ids"),
            gf_tk.get_str_arr("qwen3-tts.codec.speaker_dialects"),
        ) {
            for ((n, &id), d) in names.iter().zip(ids.iter()).zip(dialects.iter()) {
                speakers.push(SpeakerEntry {
                    name: n.to_ascii_lowercase(),
                    id,
                    dialect: d.to_ascii_lowercase(),
                });
            }
        }

        let mut owned_vram = Vec::<VramBuffer>::new();
        let mut total_vram_bytes = 0u64;
        let mut alloc_track = |gpu: &mut NvGpu, sz: u64| -> Result<VramBuffer, String> {
            let b = gpu.alloc_vram(sz)?;
            total_vram_bytes += b.size;
            Ok(b)
        };

        // ---------------------------------------------------------------------
        // 1. Upload Talker + CodePredictor Weights into Consolidated VRAM Slabs
        // ---------------------------------------------------------------------
        let mut tk_f32_slab = ByteSlab::with_capacity(8 << 20);

        let tk_kv_alloc_ctx = talker_max_ctx + TALKER_PREFILL_CHUNK;
        let tk_freqs = precompute_freqs_cis_f32(talker_head_dim, tk_kv_alloc_ctx, talker_rope_theta);
        let tk_freqs_off = tk_f32_slab.push_f32(&tk_freqs);

        let cp_freqs = precompute_freqs_cis_f32(cp_head_dim, 32, cp_rope_theta);
        let cp_freqs_off = tk_f32_slab.push_f32(&cp_freqs);

        // Talker embeddings & projections (`text_emb_dim` is 2048 even on 0.6B where `talker_dim` is 1024!)
        let (tx_emb_cow, talker_text_emb_type, _, talker_text_emb_dim) =
            extract_quant_weight_bytes(&gf_tk, "talker.text_embd.weight")?;
        let tx_emb_buf = alloc_track(gpu, tx_emb_cow.len() as u64)?;
        gpu.upload_to_vram(&tx_emb_buf, 0, &tx_emb_cow)?;
        let talker_text_emb_va = tx_emb_buf.va_addr;
        owned_vram.push(tx_emb_buf);

        let mut misc_q_slab = ByteSlab::with_capacity(256 << 20);
        let (fc1_w_cow, talker_fc1_type, talker_fc1_out_dim, _) =
            extract_quant_weight_bytes(&gf_tk, "talker.text_proj.fc1.weight")?;
        let fc1_w_off = misc_q_slab.push_bytes(&fc1_w_cow);
        let fc1_b_off = tk_f32_slab.push_f32(
            &decode_1d_f32(&gf_tk, "talker.text_proj.fc1.bias")?.ok_or("Missing fc1.bias")?,
        );

        let (fc2_w_cow, talker_fc2_type, _, _) = extract_quant_weight_bytes(&gf_tk, "talker.text_proj.fc2.weight")?;
        let fc2_w_off = misc_q_slab.push_bytes(&fc2_w_cow);
        let fc2_b_off = tk_f32_slab.push_f32(
            &decode_1d_f32(&gf_tk, "talker.text_proj.fc2.bias")?.ok_or("Missing fc2.bias")?,
        );

        let (tk_ce_cow, talker_codec_emb_type, _, _) = extract_quant_weight_bytes(&gf_tk, "talker.codec_embd.weight")?;
        let tk_ce_off = misc_q_slab.push_bytes(&tk_ce_cow);

        let tk_out_norm_off = tk_f32_slab.push_f32(
            &decode_1d_f32(&gf_tk, "talker.output_norm.weight")?.ok_or("Missing talker.output_norm.weight")?,
        );
        let (tk_ch_cow, talker_codec_head_type, _, _) = extract_quant_weight_bytes(&gf_tk, "talker.codec_head.weight")?;
        let tk_ch_off = misc_q_slab.push_bytes(&tk_ch_cow);

        // CodePredictor mtp_proj + 15 codebooks + 15 lm_heads + output_norm
        let (mtp_w_off_opt, mtp_proj_type, mtp_b_off_opt) = if gf_tk.has_tensor("code_pred.mtp_proj.weight") {
            let (mw, mt, _, _) = extract_quant_weight_bytes(&gf_tk, "code_pred.mtp_proj.weight")?;
            let woff = misc_q_slab.push_bytes(&mw);
            let boff = decode_1d_f32(&gf_tk, "code_pred.mtp_proj.bias")?
                .map(|b| tk_f32_slab.push_f32(&b));
            (Some(woff), mt, boff)
        } else {
            (None, 0, None)
        };

        let cp_out_norm_off = tk_f32_slab.push_f32(
            &decode_1d_f32(&gf_tk, "code_pred.output_norm.weight")?.ok_or("Missing code_pred.output_norm.weight")?,
        );

        let mut cp_lm_offs = [0u64; 16];
        let mut cp_emb_offs = [0u64; 16];
        let mut cp_lm_type = 8i32;
        let mut cp_emb_type = 8i32;
        for g in 0..15 {
            let (lw, lt, _, _) = extract_quant_weight_bytes(&gf_tk, &format!("code_pred.lm_head.{}.weight", g))?;
            cp_lm_offs[g] = misc_q_slab.push_bytes(&lw);
            cp_lm_type = lt;

            let (ew, et, _, _) = extract_quant_weight_bytes(&gf_tk, &format!("code_pred.codec_embd.{}.weight", g))?;
            cp_emb_offs[g] = misc_q_slab.push_bytes(&ew);
            cp_emb_type = et;
        }

        // Helper to load a Qwen3 block stack (`talker.blk.{i}` or `code_pred.blk.{i}`) into a single VRAM slab!
        struct PendingLayer {
            an_off: u64,
            fn_off: u64,
            qn_off: u64,
            kn_off: u64,
            qkv_off: u64,
            o_off: u64,
            gu_off: u64,
            dn_off: u64,
            qkv_t: i32,
            o_t: i32,
            gu_t: i32,
            dn_t: i32,
        }

        let load_stack = |prefix: &str, n_layers: usize, w_slab: &mut ByteSlab, f_slab: &mut ByteSlab| -> Result<Vec<PendingLayer>, String> {
            let mut out = Vec::with_capacity(n_layers);
            for l in 0..n_layers {
                let p = format!("{}.blk.{}", prefix, l);
                let an_off = f_slab.push_f32(&decode_1d_f32(&gf_tk, &format!("{}.attn_norm.weight", p))?.ok_or("Missing attn_norm")?);
                let fn_off = f_slab.push_f32(&decode_1d_f32(&gf_tk, &format!("{}.ffn_norm.weight", p))?.ok_or("Missing ffn_norm")?);
                let qn_off = f_slab.push_f32(&decode_1d_f32(&gf_tk, &format!("{}.attn_q_norm.weight", p))?.ok_or("Missing attn_q_norm")?);
                let kn_off = f_slab.push_f32(&decode_1d_f32(&gf_tk, &format!("{}.attn_k_norm.weight", p))?.ok_or("Missing attn_k_norm")?);

                let (wq, tq, _, _) = extract_quant_weight_bytes(&gf_tk, &format!("{}.attn_q.weight", p))?;
                let (wk, tk, _, _) = extract_quant_weight_bytes(&gf_tk, &format!("{}.attn_k.weight", p))?;
                let (wv, tv, _, _) = extract_quant_weight_bytes(&gf_tk, &format!("{}.attn_v.weight", p))?;
                if tq != tk || tk != tv {
                    return Err(format!("Mismatched QKV quant types on {}: {}, {}, {}", p, tq, tk, tv));
                }
                let mut fused_qkv = Vec::with_capacity(wq.len() + wk.len() + wv.len());
                fused_qkv.extend_from_slice(&wq);
                fused_qkv.extend_from_slice(&wk);
                fused_qkv.extend_from_slice(&wv);
                let qkv_off = w_slab.push_bytes(&fused_qkv);

                let (wo, o_t, _, _) = extract_quant_weight_bytes(&gf_tk, &format!("{}.attn_output.weight", p))?;
                let o_off = w_slab.push_bytes(&wo);

                let (wg, tg, _, _) = extract_quant_weight_bytes(&gf_tk, &format!("{}.ffn_gate.weight", p))?;
                let (wu, tu, _, _) = extract_quant_weight_bytes(&gf_tk, &format!("{}.ffn_up.weight", p))?;
                if tg != tu {
                    return Err(format!("Mismatched Gate/Up quant types on {}", p));
                }
                let mut fused_gu = Vec::with_capacity(wg.len() + wu.len());
                fused_gu.extend_from_slice(&wg);
                fused_gu.extend_from_slice(&wu);
                let gu_off = w_slab.push_bytes(&fused_gu);

                let (wd, dn_t, _, _) = extract_quant_weight_bytes(&gf_tk, &format!("{}.ffn_down.weight", p))?;
                let dn_off = w_slab.push_bytes(&wd);

                out.push(PendingLayer {
                    an_off,
                    fn_off,
                    qn_off,
                    kn_off,
                    qkv_off,
                    o_off,
                    gu_off,
                    dn_off,
                    qkv_t: tq,
                    o_t,
                    gu_t: tg,
                    dn_t,
                });
            }
            Ok(out)
        };

        let mut tk_layers_slab = ByteSlab::with_capacity(1600 << 20);
        let tk_pending = load_stack("talker", talker_num_layers, &mut tk_layers_slab, &mut tk_f32_slab)?;
        let cp_pending = load_stack("code_pred", cp_num_layers, &mut misc_q_slab, &mut tk_f32_slab)?;

        let tk_layers_vram = alloc_track(gpu, tk_layers_slab.as_slice().len() as u64)?;
        gpu.upload_to_vram(&tk_layers_vram, 0, tk_layers_slab.as_slice())?;
        let tk_w_va = tk_layers_vram.va_addr;
        owned_vram.push(tk_layers_vram);
        drop(tk_layers_slab);

        let misc_q_vram = alloc_track(gpu, misc_q_slab.as_slice().len() as u64)?;
        gpu.upload_to_vram(&misc_q_vram, 0, &misc_q_slab.as_slice())?;
        let misc_q_va = misc_q_vram.va_addr;
        owned_vram.push(misc_q_vram);
        drop(misc_q_slab);

        let tk_f32_vram = alloc_track(gpu, tk_f32_slab.as_slice().len() as u64)?;
        gpu.upload_to_vram(&tk_f32_vram, 0, &tk_f32_slab.as_slice())?;
        let tk_f32_va = tk_f32_vram.va_addr;
        owned_vram.push(tk_f32_vram);
        drop(tk_f32_slab);

        // Talker & CodePredictor KV caches
        let tk_kv_layer_bytes = ((2 * talker_n_kv_heads * tk_kv_alloc_ctx * talker_head_dim * 2) as u64 + 255) & !255;
        let tk_kv_slab = alloc_track(gpu, tk_kv_layer_bytes * (talker_num_layers as u64))?;
        let tk_kv_va = tk_kv_slab.va_addr;
        owned_vram.push(tk_kv_slab);

        let cp_kv_layer_bytes = ((2 * cp_n_kv_heads * 32 * cp_head_dim * 2) as u64 + 255) & !255;
        let cp_kv_slab = alloc_track(gpu, cp_kv_layer_bytes * (cp_num_layers as u64))?;
        let cp_kv_va = cp_kv_slab.va_addr;
        owned_vram.push(cp_kv_slab);

        let mut talker_layers = Vec::with_capacity(talker_num_layers);
        for (i, pl) in tk_pending.iter().enumerate() {
            talker_layers.push(QwenLayerDesc {
                attn_norm_w: tk_f32_va + pl.an_off,
                ffn_norm_w: tk_f32_va + pl.fn_off,
                q_norm_w: tk_f32_va + pl.qn_off,
                k_norm_w: tk_f32_va + pl.kn_off,
                w_qkv_or_qk: tk_w_va + pl.qkv_off,
                w_o: tk_w_va + pl.o_off,
                w_gate_up: tk_w_va + pl.gu_off,
                w_down: tk_w_va + pl.dn_off,
                kv_cache: tk_kv_va + (i as u64) * tk_kv_layer_bytes,
                qkv_type: pl.qkv_t,
                o_type: pl.o_t,
                gate_up_type: pl.gu_t,
                down_type: pl.dn_t,
                ..Default::default()
            });
        }
        let tk_ld_bytes = unsafe {
            std::slice::from_raw_parts(
                talker_layers.as_ptr() as *const u8,
                talker_layers.len() * std::mem::size_of::<QwenLayerDesc>(),
            )
        };
        let tk_ld_gpu = alloc_track(gpu, tk_ld_bytes.len() as u64)?;
        gpu.upload_to_vram(&tk_ld_gpu, 0, tk_ld_bytes)?;
        let tk_ld_va = tk_ld_gpu.va_addr;
        owned_vram.push(tk_ld_gpu);

        let mut cp_layers = Vec::with_capacity(cp_num_layers);
        for (i, pl) in cp_pending.iter().enumerate() {
            cp_layers.push(QwenLayerDesc {
                attn_norm_w: tk_f32_va + pl.an_off,
                ffn_norm_w: tk_f32_va + pl.fn_off,
                q_norm_w: tk_f32_va + pl.qn_off,
                k_norm_w: tk_f32_va + pl.kn_off,
                w_qkv_or_qk: misc_q_va + pl.qkv_off,
                w_o: misc_q_va + pl.o_off,
                w_gate_up: misc_q_va + pl.gu_off,
                w_down: misc_q_va + pl.dn_off,
                kv_cache: cp_kv_va + (i as u64) * cp_kv_layer_bytes,
                qkv_type: pl.qkv_t,
                o_type: pl.o_t,
                gate_up_type: pl.gu_t,
                down_type: pl.dn_t,
                ..Default::default()
            });
        }
        let cp_ld_bytes = unsafe {
            std::slice::from_raw_parts(
                cp_layers.as_ptr() as *const u8,
                cp_layers.len() * std::mem::size_of::<QwenLayerDesc>(),
            )
        };
        let cp_ld_gpu = alloc_track(gpu, cp_ld_bytes.len() as u64)?;
        gpu.upload_to_vram(&cp_ld_gpu, 0, cp_ld_bytes)?;
        let cp_ld_va = cp_ld_gpu.va_addr;
        owned_vram.push(cp_ld_gpu);

        // Talker + CodePredictor scratch workspaces
        let tk_q_dim = talker_n_heads * talker_head_dim;
        let tk_kv_dim = talker_n_kv_heads * talker_head_dim;
        let tk_proj_max_dim = talker_dim.max(talker_text_emb_dim).max(talker_fc1_out_dim);
        let tk_max_k = tk_proj_max_dim.max(tk_q_dim).max(talker_hidden_dim);

        let tk_prefill_emb = alloc_track(gpu, (1024 * tk_proj_max_dim * 4) as u64)?;
        let tk_proj_buf = alloc_track(gpu, (1024 * tk_proj_max_dim * 4) as u64)?;
        let tk_tts_pad_emb = alloc_track(gpu, (talker_dim * 4) as u64)?;
        let tk_ext_embed = alloc_track(gpu, (talker_dim * 4) as u64)?;
        let tk_hidden_last = alloc_track(gpu, (talker_dim * 4) as u64)?;
        let tk_logits_out = alloc_track(gpu, (talker_vocab * 4) as u64)?;

        let tk_ws_x_res = alloc_track(gpu, (TALKER_PREFILL_CHUNK * talker_dim * 4) as u64)?;
        let tk_ws_x_res2 = alloc_track(gpu, (TALKER_PREFILL_CHUNK * talker_dim * 4) as u64)?;
        let tk_ws_xq = alloc_track(gpu, (1024 * tk_proj_max_dim).max(TALKER_PREFILL_CHUNK * tk_max_k) as u64)?;
        let tk_ws_xds = alloc_track(gpu, ((1024 * (tk_proj_max_dim / 32) * 8).max(TALKER_PREFILL_CHUNK * (tk_max_k / 32) * 8)) as u64)?;
        let tk_ws_qkv = alloc_track(gpu, (TALKER_PREFILL_CHUNK * (tk_q_dim + 2 * tk_kv_dim) * 4) as u64)?;
        let tk_ws_v_sep = alloc_track(gpu, 4096)?;
        let tk_ws_q_roped = alloc_track(gpu, (TALKER_PREFILL_CHUNK * tk_q_dim * 4) as u64)?;
        let tk_ws_attn_out = alloc_track(gpu, (TALKER_PREFILL_CHUNK * tk_q_dim * 4) as u64)?;
        let tk_ws_gate_up = alloc_track(gpu, (TALKER_PREFILL_CHUNK * 2 * talker_hidden_dim * 4) as u64)?;
        let tk_ws_bar_state = alloc_track(gpu, 4096)?;
        gpu.zero_vram(&tk_ws_bar_state)?;
        let tk_history_seen = alloc_track(gpu, 4096)?;
        gpu.zero_vram(&tk_history_seen)?;

        let tk_cta_dummy = alloc_track(gpu, 4096)?;
        let tk_cta_va = tk_cta_dummy.va_addr;
        owned_vram.push(tk_cta_dummy);

        let talker_freqs_va = tk_f32_va + tk_freqs_off;

        let tk_mdesc = QwenModelDesc {
            layers_ptr: tk_ld_va,
            emb_w: misc_q_va + tk_ce_off,
            output_norm_w: tk_f32_va + tk_out_norm_off,
            lm_head_w: misc_q_va + tk_ch_off,
            freqs_cis: tk_f32_va + tk_freqs_off,
            ws_x_res: tk_ws_x_res.va_addr,
            ws_x_res2: tk_ws_x_res2.va_addr,
            ws_xq: tk_ws_xq.va_addr,
            ws_xds: tk_ws_xds.va_addr,
            ws_qkv: tk_ws_qkv.va_addr,
            ws_q_roped: tk_ws_q_roped.va_addr,
            ws_gate_up: tk_ws_gate_up.va_addr,
            ws_cta_val: tk_cta_va,
            ws_cta_idx: tk_cta_va + 2048,
            ws_bar_state: tk_ws_bar_state.va_addr,
            num_layers: talker_num_layers as i32,
            dim: talker_dim as i32,
            hidden_dim: talker_hidden_dim as i32,
            n_heads: talker_n_heads as i32,
            n_kv_heads: talker_n_kv_heads as i32,
            head_dim: talker_head_dim as i32,
            rope_dim: talker_head_dim as i32,
            vocab_size: talker_vocab as i32,
            max_ctx: tk_kv_alloc_ctx as i32,
            has_qk_norm: 1,
            emb_type: -1, // Read external FP32 embedding from `tk_ext_embed`!
            lm_head_type: talker_codec_head_type,
            norm_eps: talker_eps,
            inv_sqrt_head_dim: 1.0 / (talker_head_dim as f32).sqrt(),
            stop_id0: codec_eos_id,
            stop_id1: codec_eos_id,
            stop_id2: codec_eos_id,
            stop_id3: codec_eos_id,
            ws_ext_embed: tk_ext_embed.va_addr,
            ws_hidden_last: tk_hidden_last.va_addr,
            ws_logits_out: tk_logits_out.va_addr,
            _pad_model: 0,
        };
        let tk_mdesc_bytes = unsafe {
            std::slice::from_raw_parts(
                &tk_mdesc as *const QwenModelDesc as *const u8,
                std::mem::size_of::<QwenModelDesc>(),
            )
        };
        let talker_model_desc_gpu = alloc_track(gpu, 4096)?;
        gpu.upload_to_vram(&talker_model_desc_gpu, 0, tk_mdesc_bytes)?;

        // CodePredictor scratch slab
        let cp_ws_slab = alloc_track(gpu, 256 * 1024)?;
        gpu.zero_vram(&cp_ws_slab)?;
        let cp_ws_base = cp_ws_slab.va_addr;
        owned_vram.push(cp_ws_slab);

        let cp_ws_codes_frame = alloc_track(gpu, 4096)?;
        gpu.zero_vram(&cp_ws_codes_frame)?;
        let cp_ws_bar_state = alloc_track(gpu, 4096)?;
        gpu.zero_vram(&cp_ws_bar_state)?;

        let mut cp_lm_vas = [0u64; 16];
        let mut cp_emb_vas = [0u64; 16];
        for g in 0..15 {
            cp_lm_vas[g] = misc_q_va + cp_lm_offs[g];
            cp_emb_vas[g] = misc_q_va + cp_emb_offs[g];
        }

        let cp_desc = TtsCodePredDesc {
            layers_ptr: cp_ld_va,
            mtp_proj_w: mtp_w_off_opt.map_or(0, |o| misc_q_va + o),
            mtp_proj_b: mtp_b_off_opt.map_or(0, |o| tk_f32_va + o),
            output_norm_w: tk_f32_va + cp_out_norm_off,
            lm_head_w: cp_lm_vas,
            cp_codec_emb_w: cp_emb_vas,
            talker_codec_emb_w: misc_q_va + tk_ce_off,
            freqs_cis: tk_f32_va + cp_freqs_off,
            talker_hidden_last: tk_hidden_last.va_addr,
            tts_pad_embed: tk_tts_pad_emb.va_addr,
            talker_ws_ext_embed: tk_ext_embed.va_addr,
            ws_in2048: cp_ws_base + 0,
            ws_x_res: cp_ws_base + 16384,
            ws_x_res2: cp_ws_base + 32768,
            ws_xq: cp_ws_base + 49152,
            ws_xds: cp_ws_base + 65536,
            ws_qkv: cp_ws_base + 81920,
            ws_q_roped: cp_ws_base + 114688,
            ws_gate_up: cp_ws_base + 131072,
            ws_logits: cp_ws_base + 180224,
            ws_codes_frame: cp_ws_codes_frame.va_addr,
            ws_bar_state: cp_ws_bar_state.va_addr,
            num_layers: cp_num_layers as i32,
            talker_dim: talker_dim as i32,
            dim: cp_dim as i32,
            hidden_dim: cp_hidden_dim as i32,
            n_heads: cp_n_heads as i32,
            n_kv_heads: cp_n_kv_heads as i32,
            head_dim: cp_head_dim as i32,
            vocab_size: cp_vocab as i32,
            max_ctx: 32,
            mtp_proj_type,
            talker_codec_emb_type,
            lm_head_type: cp_lm_type,
            cp_codec_emb_type: cp_emb_type,
            norm_eps: cp_eps,
            inv_sqrt_head_dim: 1.0 / (cp_head_dim as f32).sqrt(),
            _pad_cp: 0,
        };
        let cp_desc_bytes = unsafe {
            std::slice::from_raw_parts(
                &cp_desc as *const TtsCodePredDesc as *const u8,
                std::mem::size_of::<TtsCodePredDesc>(),
            )
        };
        let cp_desc_gpu = alloc_track(gpu, 4096)?;
        gpu.upload_to_vram(&cp_desc_gpu, 0, cp_desc_bytes)?;

        // ---------------------------------------------------------------------
        // 2. Load 100%-on-RTX-3080 `12Hz Code2Wav` Neural Vocoder (`tok_dec.*`)
        // ---------------------------------------------------------------------
        let c2w = Code2WavEngine::load(gpu, &gf_cd, &mut total_vram_bytes)?;

        // 3 Host-mapped control/audio buffers (`MAP_SYSMEM_FD` budget = only 3 slots!)
        let tk_decode_ctrl = gpu.alloc_host_mapped(std::mem::size_of::<QwenDecodeCtrl>())?;
        let frame_ctrl = gpu.alloc_host_mapped(std::mem::size_of::<TtsFrameCtrl>())?;
        // 16 KB staging at `[0..16384)`, plus `TTS_MAX_CODEC_CHUNK_FRAMES * 1920 * 4` (`245,760 B`) audio out at `[16384..]`
        let host_io_buf = gpu.alloc_host_mapped(16384 + TTS_MAX_CODEC_CHUNK_FRAMES * TTS_HOP_SAMPLES * 4)?;

        let total_vram_mb = (total_vram_bytes as f64) / (1024.0 * 1024.0);
        eprintln!(
            "[qwen-nv tts] Loaded '{}' + 12Hz Code2Wav ({:.1} MB VRAM, {} voices, 100% RTX 3080 sm_86) in {:.2}s",
            model_name,
            total_vram_mb,
            speakers.len(),
            t0.elapsed().as_secs_f64()
        );

        Ok(Self {
            talker_ref: talker_ref.to_string(),
            model_name,
            tok,
            _owned_vram: owned_vram,
            speakers,
            languages,
            codec_pad_id,
            codec_bos_id,
            codec_eos_id,
            codec_think_id,
            codec_nothink_id,
            codec_think_bos_id,
            codec_think_eos_id,
            tts_pad_id,
            tts_bos_id,
            tts_eos_id,
            talker_dim,
            talker_text_emb_dim,
            talker_fc1_out_dim,
            talker_hidden_dim,
            talker_n_heads,
            talker_n_kv_heads,
            talker_head_dim,
            talker_vocab,
            talker_max_ctx,
            talker_eps,
            talker_layers,
            talker_model_desc_gpu,
            talker_freqs_va,
            talker_text_emb_va,
            talker_text_emb_type,
            talker_fc1_w: misc_q_va + fc1_w_off,
            talker_fc1_b: tk_f32_va + fc1_b_off,
            talker_fc1_type,
            talker_fc2_w: misc_q_va + fc2_w_off,
            talker_fc2_b: tk_f32_va + fc2_b_off,
            talker_fc2_type,
            talker_codec_emb_va: misc_q_va + tk_ce_off,
            talker_codec_emb_type,
            talker_out_norm_va: tk_f32_va + tk_out_norm_off,
            talker_codec_head_va: misc_q_va + tk_ch_off,
            talker_codec_head_type,
            tk_prefill_emb,
            tk_proj_buf,
            tk_tts_pad_emb,
            tk_ext_embed,
            tk_hidden_last,
            tk_logits_out,
            tk_ws_x_res,
            tk_ws_x_res2,
            tk_ws_xq,
            tk_ws_xds,
            tk_ws_qkv,
            tk_ws_v_sep,
            tk_ws_q_roped,
            tk_ws_attn_out,
            tk_ws_gate_up,
            tk_ws_bar_state,
            tk_history_seen,
            cp_desc_gpu,
            cp_ws_codes_frame,
            cp_ws_bar_state,
            tk_decode_ctrl,
            frame_ctrl,
            host_io_buf,
            c2w,
            total_vram_mb,
        })
    }

    pub fn speaker_names(&self) -> Vec<&str> {
        self.speakers.iter().map(|s| s.name.as_str()).collect()
    }

    pub fn reset_codec_stream(&mut self, gpu: &mut NvGpu) -> Result<(), String> {
        self.c2w.reset_stream(gpu)
    }

    fn execute_qmd_chain(gpu: &mut NvGpu, chain: &[(u64, usize)], timeout_secs: u64) -> Result<(), String> {
        gpu.launch_qmd_chain_sync(chain, true, Duration::from_secs(timeout_secs))
    }

    pub fn decode_codes_stream_chunk(
        &mut self,
        gpu: &mut NvGpu,
        module: &CubinModule,
        codes_flat: &[i32],
        n_frames: usize,
    ) -> Result<Vec<f32>, String> {
        self.c2w.decode_stream_chunk(gpu, module, &mut self.host_io_buf, codes_flat, n_frames)
    }

    /// Build Talker `[T_ctx, 2048]` prompt embeddings & run Talker Prefill 100% on RTX 3080!
    fn build_and_prefill_talker(
        &mut self,
        gpu: &mut NvGpu,
        module: &CubinModule,
        params: &TtsSynthParams,
    ) -> Result<(usize, String, String), String> {
        // Resolve language & speaker (with dialect override matching `prompt-builder.h:461-483`)
        let lang_req = if params.language.trim().is_empty() {
            "english".to_string()
        } else {
            params.language.trim().to_ascii_lowercase()
        };
        let mut lang_id: Option<i32> = if lang_req == "auto" {
            None
        } else {
            Some(
                self.languages
                    .iter()
                    .find(|l| l.name == lang_req)
                    .map(|l| l.id)
                    .ok_or_else(|| format!("Unknown TTS language '{}'", params.language))?,
            )
        };

        let spk_req = params.speaker.trim().to_ascii_lowercase();
        let mut spk_id: Option<i32> = None;
        let mut eff_spk_name = "none".to_string();
        if !spk_req.is_empty() && spk_req != "none" && !self.speakers.is_empty() {
            let found = self
                .speakers
                .iter()
                .find(|s| s.name == spk_req)
                .ok_or_else(|| {
                    let avail: Vec<&str> = self.speakers.iter().map(|s| s.name.as_str()).collect();
                    format!("Unknown speaker '{}' (available: {})", params.speaker, avail.join(", "))
                })?;
            spk_id = Some(found.id);
            eff_spk_name = found.name.clone();
            if !found.dialect.is_empty() && (lang_req == "chinese" || lang_req == "auto") {
                if let Some(dl) = self.languages.iter().find(|l| l.name == found.dialect) {
                    lang_id = Some(dl.id);
                }
            }
        }

        let full_text = format!("<|im_start|>assistant\n{}<|im_end|>\n<|im_start|>assistant\n", params.text);
        let ids = self.tok.encode(&full_text);
        if ids.len() < 9 {
            return Err(format!("Utterance tokenized too short ({} tokens)", ids.len()));
        }
        let n_text = ids.len() - 8;

        let instruct_ids = if !params.instruct.trim().is_empty() {
            let wrapped = format!("<|im_start|>user\n{}<|im_end|>\n", params.instruct.trim());
            self.tok.encode(&wrapped)
        } else {
            Vec::new()
        };

        let mut codec_left = Vec::<i32>::with_capacity(8);
        if let Some(lid) = lang_id {
            codec_left.extend_from_slice(&[
                self.codec_think_id,
                self.codec_think_bos_id,
                lid,
                self.codec_think_eos_id,
            ]);
        } else {
            codec_left.extend_from_slice(&[
                self.codec_nothink_id,
                self.codec_think_bos_id,
                self.codec_think_eos_id,
            ]);
        }
        if let Some(sid) = spk_id {
            codec_left.push(sid);
        }
        codec_left.push(self.codec_pad_id);

        let n_head = instruct_ids.len() + 3;
        let n_codec_left = codec_left.len();
        let t_ctx = n_head + n_codec_left + n_text + 2;
        if t_ctx >= self.talker_max_ctx || 3 + n_head + n_text > 1024 {
            return Err(format!("Utterance too long ({} rows)", t_ctx));
        }

        // Pack `[tts_bos, tts_eos, tts_pad] + head_ids + text_ids` + `codec_left` into `host_io_buf[0..16KB]`
        let total_proj = 3 + n_head + n_text;
        let codec_off_bytes = ((total_proj * 4) + 63) & !63;
        {
            let dst = self.host_io_buf.as_mut_slice();
            let mut all_tx = Vec::with_capacity(total_proj);
            all_tx.push(self.tts_bos_id);
            all_tx.push(self.tts_eos_id);
            all_tx.push(self.tts_pad_id);
            all_tx.extend_from_slice(&instruct_ids);
            all_tx.extend_from_slice(&ids[0..3]);
            all_tx.extend_from_slice(&ids[3..3 + n_text]);

            for (i, &id) in all_tx.iter().enumerate() {
                put_u32(dst, i * 4, id as u32);
            }
            for (i, &cid) in codec_left.iter().enumerate() {
                put_u32(dst, codec_off_bytes + i * 4, cid as u32);
            }
        }
        fence(Ordering::SeqCst);

        // 1. Build `[t_ctx, 2048]` embeddings in VRAM via `text_embd -> fc1 -> SiLU -> fc2 -> assemble`!
        {
            let k_emb = module.kernel("qwen_prefill_embed")?;
            let k_q8 = module.kernel("qwen_prefill_quant_q8_1")?;
            let k_gemm = module.kernel("qwen_prefill_gemm")?;
            let k_silu = module.kernel("tts_silu_inplace_f32")?;
            let k_asm = module.kernel("tts_assemble_prefill_embed")?;

            let mut chain = Vec::with_capacity(8);
            let dim = self.talker_dim as u32;
            let tx_dim = self.talker_text_emb_dim as u32;
            let fc1_dim = self.talker_fc1_out_dim as u32;
            let m_u = total_proj as u32;
            let grid_m = (m_u + 7) / 8;

            // Embed `[total_proj, text_emb_dim]` into `tk_prefill_emb` (temp)
            let mut a_e = [0u8; 32];
            put_u64(&mut a_e, 0, self.tk_prefill_emb.va_addr);
            put_u64(&mut a_e, 8, self.talker_text_emb_va);
            put_u64(&mut a_e, 16, self.host_io_buf.va_addr);
            put_u32(&mut a_e, 24, tx_dim);
            put_u32(&mut a_e, 28, self.talker_text_emb_type as u32);
            chain.push(gpu.prepare_qmd(k_emb, (m_u, 1, 1), (256, 1, 1), &a_e, None));

            let blks1 = ((total_proj * self.talker_text_emb_dim) / 32) as u32;
            let mut a_q1 = [0u8; 28];
            put_u64(&mut a_q1, 0, self.tk_ws_xq.va_addr);
            put_u64(&mut a_q1, 8, self.tk_ws_xds.va_addr);
            put_u64(&mut a_q1, 16, self.tk_prefill_emb.va_addr);
            put_u32(&mut a_q1, 24, blks1);
            chain.push(gpu.prepare_qmd(k_q8, ((blks1 + 3) / 4, 1, 1), (128, 1, 1), &a_q1, None));

            // fc1 (`text_emb_dim -> fc1_out_dim`) -> `tk_proj_buf`
            let mut a_fc1 = [0u8; 64];
            put_u64(&mut a_fc1, 0, self.tk_proj_buf.va_addr);
            put_u64(&mut a_fc1, 8, self.talker_fc1_w);
            put_u64(&mut a_fc1, 16, self.tk_ws_xq.va_addr);
            put_u64(&mut a_fc1, 24, self.tk_ws_xds.va_addr);
            put_u64(&mut a_fc1, 32, self.talker_fc1_b);
            put_u64(&mut a_fc1, 40, 0);
            put_u32(&mut a_fc1, 48, m_u);
            put_u32(&mut a_fc1, 52, fc1_dim);
            put_u32(&mut a_fc1, 56, tx_dim);
            put_u32(&mut a_fc1, 60, self.talker_fc1_type as u32);
            chain.push(gpu.prepare_qmd(k_gemm, ((fc1_dim + 3) / 4, grid_m, 1), (128, 1, 1), &a_fc1, None));

            let nel = (total_proj * self.talker_fc1_out_dim) as u32;
            let mut a_si = [0u8; 12];
            put_u64(&mut a_si, 0, self.tk_proj_buf.va_addr);
            put_u32(&mut a_si, 8, nel);
            chain.push(gpu.prepare_qmd(k_silu, ((nel + 255) / 256, 1, 1), (256, 1, 1), &a_si, None));

            let blks2 = ((total_proj * self.talker_fc1_out_dim) / 32) as u32;
            let mut a_q2 = [0u8; 28];
            put_u64(&mut a_q2, 0, self.tk_ws_xq.va_addr);
            put_u64(&mut a_q2, 8, self.tk_ws_xds.va_addr);
            put_u64(&mut a_q2, 16, self.tk_proj_buf.va_addr);
            put_u32(&mut a_q2, 24, blks2);
            chain.push(gpu.prepare_qmd(k_q8, ((blks2 + 3) / 4, 1, 1), (128, 1, 1), &a_q2, None));

            // fc2 (`fc1_out_dim -> talker_dim`) -> `tk_proj_buf` (safe because input is in `tk_ws_xq`)
            let mut a_fc2 = [0u8; 64];
            put_u64(&mut a_fc2, 0, self.tk_proj_buf.va_addr);
            put_u64(&mut a_fc2, 8, self.talker_fc2_w);
            put_u64(&mut a_fc2, 16, self.tk_ws_xq.va_addr);
            put_u64(&mut a_fc2, 24, self.tk_ws_xds.va_addr);
            put_u64(&mut a_fc2, 32, self.talker_fc2_b);
            put_u64(&mut a_fc2, 40, 0);
            put_u32(&mut a_fc2, 48, m_u);
            put_u32(&mut a_fc2, 52, dim);
            put_u32(&mut a_fc2, 56, fc1_dim);
            put_u32(&mut a_fc2, 60, self.talker_fc2_type as u32);
            chain.push(gpu.prepare_qmd(k_gemm, ((dim + 3) / 4, grid_m, 1), (128, 1, 1), &a_fc2, None));

            // Assemble `[t_ctx, 2048]` into `tk_prefill_emb` + `tk_tts_pad_emb`
            let mut a_as = [0u8; 68];
            put_u64(&mut a_as, 0, self.tk_prefill_emb.va_addr);
            put_u64(&mut a_as, 8, self.tk_tts_pad_emb.va_addr);
            put_u64(&mut a_as, 16, self.tk_proj_buf.va_addr);
            put_u64(&mut a_as, 24, self.talker_codec_emb_va);
            put_u64(&mut a_as, 32, self.host_io_buf.va_addr + (codec_off_bytes as u64));
            put_u32(&mut a_as, 40, n_head as u32);
            put_u32(&mut a_as, 44, n_codec_left as u32);
            put_u32(&mut a_as, 48, n_text as u32);
            put_u32(&mut a_as, 52, self.codec_pad_id as u32);
            put_u32(&mut a_as, 56, self.codec_bos_id as u32);
            put_u32(&mut a_as, 60, dim);
            put_u32(&mut a_as, 64, self.talker_codec_emb_type as u32);
            chain.push(gpu.prepare_qmd(k_asm, (t_ctx as u32, 1, 1), (256, 1, 1), &a_as, None));

            Self::execute_qmd_chain(gpu, &chain, 15)?;
        }

        // 2. Run Talker 28-Layer Prefill over `[0 .. t_ctx)` -> writes KV `[0..t_ctx-1]`, `tk_hidden_last`, `tk_logits_out`!
        let mut pos = 0usize;
        while pos < t_ctx {
            let take = (t_ctx - pos).min(TALKER_PREFILL_CHUNK);
            let is_final_chunk = pos + take == t_ctx;
            self.run_talker_prefill_chunk(gpu, module, pos, take, is_final_chunk)?;
            pos += take;
        }

        Ok((t_ctx, eff_spk_name, lang_req))
    }

    fn run_talker_prefill_chunk(
        &mut self,
        gpu: &mut NvGpu,
        module: &CubinModule,
        start_pos: usize,
        t_len: usize,
        compute_final_head: bool,
    ) -> Result<(), String> {
        let k_copy = module.kernel("tts_copy_f32")?;
        let k_rms = module.kernel("qwen_prefill_rmsnorm_q8_1")?;
        let k_gemm = module.kernel("qwen_prefill_gemm")?;
        let k_qknorm = module.kernel("qwen_prefill_qknorm_rope_kv")?;
        let k_flash = module.kernel("qwen_prefill_flash_gqa")?;
        let k_quant = module.kernel("qwen_prefill_quant_q8_1")?;
        let k_swiglu = module.kernel("qwen_prefill_swiglu_q8_1")?;
        let k_out_rms = module.kernel("tts_rmsnorm_f32_q8_1")?;

        let dim = self.talker_dim as u32;
        let hidden_dim = self.talker_hidden_dim as u32;
        let n_heads = self.talker_n_heads as u32;
        let n_kv_heads = self.talker_n_kv_heads as u32;
        let head_dim = self.talker_head_dim as u32;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let qkv_n = q_dim + 2 * kv_dim;
        let m_u = t_len as u32;
        let grid_m = (m_u + 7) / 8;
        let eps_bits = self.talker_eps.to_bits();
        let scale_bits = (1.0f32 / (self.talker_head_dim as f32).sqrt()).to_bits();
        let max_ctx_u = (self.talker_max_ctx + TALKER_PREFILL_CHUNK) as u32;
        let freqs_va = unsafe {
            let mdesc = &*(self.talker_layers.as_ptr() as *const QwenLayerDesc);
            let _ = mdesc;
            // Read freqs_cis from `talker_model_desc` calculation:
            0u64
        };
        let _ = freqs_va;

        let mut chain = Vec::with_capacity(4 + self.talker_layers.len() * 8);

        // Copy pre-built embeddings `tk_prefill_emb[start_pos .. start_pos + t_len]` -> `tk_ws_x_res`
        let copy_elems = (t_len * self.talker_dim) as u32;
        let src_emb_va = self.tk_prefill_emb.va_addr + (start_pos * self.talker_dim * 4) as u64;
        let mut a_cp = [0u8; 20];
        put_u64(&mut a_cp, 0, self.tk_ws_x_res.va_addr);
        put_u64(&mut a_cp, 8, src_emb_va);
        put_u32(&mut a_cp, 16, copy_elems);
        chain.push(gpu.prepare_qmd(k_copy, ((copy_elems + 255) / 256, 1, 1), (256, 1, 1), &a_cp, None));

        let tk_freqs_va = self.talker_freqs_va;

        for l in 0..self.talker_layers.len() {
            let ld = self.talker_layers[l];
            let mut a_rms1 = [0u8; 40];
            put_u64(&mut a_rms1, 0, self.tk_ws_xq.va_addr);
            put_u64(&mut a_rms1, 8, self.tk_ws_xds.va_addr);
            put_u64(&mut a_rms1, 16, self.tk_ws_x_res.va_addr);
            put_u64(&mut a_rms1, 24, ld.attn_norm_w);
            put_u32(&mut a_rms1, 32, dim);
            put_u32(&mut a_rms1, 36, eps_bits);
            chain.push(gpu.prepare_qmd(k_rms, (m_u, 1, 1), (256, 1, 1), &a_rms1, None));

            let mut a_qk = [0u8; 64];
            put_u64(&mut a_qk, 0, self.tk_ws_qkv.va_addr);
            put_u64(&mut a_qk, 8, ld.w_qkv_or_qk);
            put_u64(&mut a_qk, 16, self.tk_ws_xq.va_addr);
            put_u64(&mut a_qk, 24, self.tk_ws_xds.va_addr);
            put_u64(&mut a_qk, 32, 0);
            put_u64(&mut a_qk, 40, 0);
            put_u32(&mut a_qk, 48, m_u);
            put_u32(&mut a_qk, 52, qkv_n);
            put_u32(&mut a_qk, 56, dim);
            put_u32(&mut a_qk, 60, ld.qkv_type as u32);
            chain.push(gpu.prepare_qmd(k_gemm, ((qkv_n + 3) / 4, grid_m, 1), (128, 1, 1), &a_qk, None));

            let mut a_rope = [0u8; 96];
            put_u64(&mut a_rope, 0, self.tk_ws_q_roped.va_addr);
            put_u64(&mut a_rope, 8, ld.kv_cache);
            put_u64(&mut a_rope, 16, self.tk_ws_qkv.va_addr);
            put_u64(&mut a_rope, 24, 0);
            put_u64(&mut a_rope, 32, ld.q_norm_w);
            put_u64(&mut a_rope, 40, ld.k_norm_w);
            put_u64(&mut a_rope, 48, tk_freqs_va);
            put_u32(&mut a_rope, 56, m_u);
            put_u32(&mut a_rope, 60, start_pos as u32);
            put_u32(&mut a_rope, 64, n_heads);
            put_u32(&mut a_rope, 68, n_kv_heads);
            put_u32(&mut a_rope, 72, head_dim);
            put_u32(&mut a_rope, 76, head_dim);
            put_u32(&mut a_rope, 80, max_ctx_u);
            put_u32(&mut a_rope, 84, eps_bits);
            put_u32(&mut a_rope, 88, 1);
            put_u32(&mut a_rope, 92, 0);
            chain.push(gpu.prepare_qmd(k_qknorm, (n_heads + 2 * n_kv_heads, m_u, 1), (256, 1, 1), &a_rope, None));

            if l + 1 == self.talker_layers.len() && !compute_final_head {
                break;
            }

            let mut a_fl = [0u8; 52];
            put_u64(&mut a_fl, 0, self.tk_ws_attn_out.va_addr);
            put_u64(&mut a_fl, 8, self.tk_ws_q_roped.va_addr);
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
            put_u64(&mut a_q8, 0, self.tk_ws_xq.va_addr);
            put_u64(&mut a_q8, 8, self.tk_ws_xds.va_addr);
            put_u64(&mut a_q8, 16, self.tk_ws_attn_out.va_addr);
            put_u32(&mut a_q8, 24, attn_blks);
            chain.push(gpu.prepare_qmd(k_quant, ((attn_blks + 3) / 4, 1, 1), (128, 1, 1), &a_q8, None));

            let mut a_wo = [0u8; 64];
            put_u64(&mut a_wo, 0, self.tk_ws_x_res2.va_addr);
            put_u64(&mut a_wo, 8, ld.w_o);
            put_u64(&mut a_wo, 16, self.tk_ws_xq.va_addr);
            put_u64(&mut a_wo, 24, self.tk_ws_xds.va_addr);
            put_u64(&mut a_wo, 32, 0);
            put_u64(&mut a_wo, 40, self.tk_ws_x_res.va_addr);
            put_u32(&mut a_wo, 48, m_u);
            put_u32(&mut a_wo, 52, dim);
            put_u32(&mut a_wo, 56, q_dim);
            put_u32(&mut a_wo, 60, ld.o_type as u32);
            chain.push(gpu.prepare_qmd(k_gemm, ((dim + 3) / 4, grid_m, 1), (128, 1, 1), &a_wo, None));

            let mut a_rms2 = [0u8; 40];
            put_u64(&mut a_rms2, 0, self.tk_ws_xq.va_addr);
            put_u64(&mut a_rms2, 8, self.tk_ws_xds.va_addr);
            put_u64(&mut a_rms2, 16, self.tk_ws_x_res2.va_addr);
            put_u64(&mut a_rms2, 24, ld.ffn_norm_w);
            put_u32(&mut a_rms2, 32, dim);
            put_u32(&mut a_rms2, 36, eps_bits);
            chain.push(gpu.prepare_qmd(k_rms, (m_u, 1, 1), (256, 1, 1), &a_rms2, None));

            let gu_n = 2 * hidden_dim;
            let mut a_gu = [0u8; 64];
            put_u64(&mut a_gu, 0, self.tk_ws_gate_up.va_addr);
            put_u64(&mut a_gu, 8, ld.w_gate_up);
            put_u64(&mut a_gu, 16, self.tk_ws_xq.va_addr);
            put_u64(&mut a_gu, 24, self.tk_ws_xds.va_addr);
            put_u64(&mut a_gu, 32, 0);
            put_u64(&mut a_gu, 40, 0);
            put_u32(&mut a_gu, 48, m_u);
            put_u32(&mut a_gu, 52, gu_n);
            put_u32(&mut a_gu, 56, dim);
            put_u32(&mut a_gu, 60, ld.gate_up_type as u32);
            chain.push(gpu.prepare_qmd(k_gemm, ((gu_n + 3) / 4, grid_m, 1), (128, 1, 1), &a_gu, None));

            let ffn_blks = ((t_len as u32) * hidden_dim) / 32;
            let mut a_sw = [0u8; 32];
            put_u64(&mut a_sw, 0, self.tk_ws_xq.va_addr);
            put_u64(&mut a_sw, 8, self.tk_ws_xds.va_addr);
            put_u64(&mut a_sw, 16, self.tk_ws_gate_up.va_addr);
            put_u32(&mut a_sw, 24, m_u);
            put_u32(&mut a_sw, 28, hidden_dim);
            chain.push(gpu.prepare_qmd(k_swiglu, ((ffn_blks + 3) / 4, 1, 1), (128, 1, 1), &a_sw, None));

            let mut a_dn = [0u8; 64];
            put_u64(&mut a_dn, 0, self.tk_ws_x_res.va_addr);
            put_u64(&mut a_dn, 8, ld.w_down);
            put_u64(&mut a_dn, 16, self.tk_ws_xq.va_addr);
            put_u64(&mut a_dn, 24, self.tk_ws_xds.va_addr);
            put_u64(&mut a_dn, 32, 0);
            put_u64(&mut a_dn, 40, self.tk_ws_x_res2.va_addr);
            put_u32(&mut a_dn, 48, m_u);
            put_u32(&mut a_dn, 52, dim);
            put_u32(&mut a_dn, 56, hidden_dim);
            put_u32(&mut a_dn, 60, ld.down_type as u32);
            chain.push(gpu.prepare_qmd(k_gemm, ((dim + 3) / 4, grid_m, 1), (128, 1, 1), &a_dn, None));
        }

        if compute_final_head {
            // Final row `t_len - 1` -> `tk_hidden_last` + `(xq, xds)` -> `codec_head` -> `tk_logits_out [3072]`!
            let last_row_va = self.tk_ws_x_res.va_addr + ((t_len - 1) * self.talker_dim * 4) as u64;
            let mut a_or = [0u8; 48];
            put_u64(&mut a_or, 0, self.tk_hidden_last.va_addr);
            put_u64(&mut a_or, 8, self.tk_ws_xq.va_addr);
            put_u64(&mut a_or, 16, self.tk_ws_xds.va_addr);
            put_u64(&mut a_or, 24, last_row_va);
            put_u64(&mut a_or, 32, self.talker_out_norm_va);
            put_u32(&mut a_or, 40, dim);
            put_u32(&mut a_or, 44, eps_bits);
            chain.push(gpu.prepare_qmd(k_out_rms, (1, 1, 1), (256, 1, 1), &a_or, None));

            let vocab_u = self.talker_vocab as u32;
            let mut a_ch = [0u8; 64];
            put_u64(&mut a_ch, 0, self.tk_logits_out.va_addr);
            put_u64(&mut a_ch, 8, self.talker_codec_head_va);
            put_u64(&mut a_ch, 16, self.tk_ws_xq.va_addr);
            put_u64(&mut a_ch, 24, self.tk_ws_xds.va_addr);
            put_u64(&mut a_ch, 32, 0);
            put_u64(&mut a_ch, 40, 0);
            put_u32(&mut a_ch, 48, 1);
            put_u32(&mut a_ch, 52, vocab_u);
            put_u32(&mut a_ch, 56, dim);
            put_u32(&mut a_ch, 60, self.talker_codec_head_type as u32);
            chain.push(gpu.prepare_qmd(k_gemm, ((vocab_u + 3) / 4, 1, 1), (128, 1, 1), &a_ch, None));
        }

        Self::execute_qmd_chain(gpu, &chain, 30)
    }

    /// Full End-to-End Streaming Synthesis (`Text -> 24 kHz PCM Audio`) 100% on RTX 3080!
    /// Streams audio chunks (`on_chunk(&[f32]) -> bool`) starting right on Frame 0 for ultra-low TTFA!
    pub fn synthesize<F>(
        &mut self,
        gpu: &mut NvGpu,
        module: &CubinModule,
        params: &TtsSynthParams,
        cancel: Option<&AtomicBool>,
        mut on_chunk: F,
    ) -> Result<(Vec<f32>, TtsStats), String>
    where
        F: FnMut(&[f32]) -> bool,
    {
        if params.text.trim().is_empty() {
            return Err("TTS input text is empty".into());
        }
        let t_total = Instant::now();
        gpu.zero_vram(&self.tk_history_seen)?;
        self.reset_codec_stream(gpu)?;

        let seed = params.seed.unwrap_or_else(|| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(42)
        });

        // 1. Prompt Embedding Build + 28L Talker Prefill on RTX 3080
        let t_pf0 = Instant::now();
        let (t_ctx, eff_spk, eff_lang) = self.build_and_prefill_talker(gpu, module, params)?;
        let prefill_ms = t_pf0.elapsed().as_secs_f64() * 1000.0;

        // Initialize `TtsFrameCtrl` & `QwenDecodeCtrl` in host-mapped memory
        let fctrl_ptr = self.frame_ctrl.as_ptr() as *mut TtsFrameCtrl;
        let tk_ctrl_ptr = self.tk_decode_ctrl.as_ptr() as *mut QwenDecodeCtrl;
        unsafe {
            (*fctrl_ptr).seed = seed;
            (*fctrl_ptr).overlay_ptr = self.tk_tts_pad_emb.va_addr;
            (*fctrl_ptr).frame_idx = 0;
            (*fctrl_ptr).c0 = 0;
            (*fctrl_ptr).is_eos = 0;
            (*fctrl_ptr).error_code = 0;
            (*fctrl_ptr).frames_completed = 0;
            (*fctrl_ptr).talker_temp = params.temperature;
            (*fctrl_ptr).talker_top_k = params.top_k;
            (*fctrl_ptr).talker_top_p = params.top_p;
            (*fctrl_ptr).rep_penalty = params.rep_penalty;
            (*fctrl_ptr).sub_temp = params.sub_temperature;
            (*fctrl_ptr).sub_top_k = params.sub_top_k;
            (*fctrl_ptr).sub_top_p = params.sub_top_p;
            (*fctrl_ptr).codec_eos_id = self.codec_eos_id;

            (*tk_ctrl_ptr).start_token = 0;
            (*tk_ctrl_ptr).start_pos = t_ctx as i32;
            (*tk_ctrl_ptr).max_steps = 1;
            (*tk_ctrl_ptr).temperature = 0.0;
            (*tk_ctrl_ptr).rng_seed = seed;
            (*tk_ctrl_ptr).cancel_flag = 0;
            (*tk_ctrl_ptr).error_code = 0;
            (*tk_ctrl_ptr).steps_completed = 0;
            (*tk_ctrl_ptr).stop_reason = 0;
            (*tk_ctrl_ptr).export_logits = 1;
        }
        fence(Ordering::SeqCst);

        let max_frames = params.max_frames.clamp(1, (2048usize).saturating_sub(t_ctx + 2));
        let num_sms = gpu.active_sms;
        let k_tk_mk = module.kernel("qwen_persistent_decode_megakernel")?;
        let k_sample_c0 = module.kernel("tts_sample_c0_kernel")?;
        let k_cp_mk = module.kernel("tts_code_predictor_frame_megakernel")?;

        let mut mk_tk_args = [0u8; 16];
        put_u64(&mut mk_tk_args, 0, self.talker_model_desc_gpu.va_addr);
        put_u64(&mut mk_tk_args, 8, self.tk_decode_ctrl.va_addr);

        let mut sample_c0_args = [0u8; 36];
        put_u64(&mut sample_c0_args, 0, self.frame_ctrl.va_addr);
        put_u64(&mut sample_c0_args, 8, self.tk_history_seen.va_addr);
        put_u64(&mut sample_c0_args, 16, self.cp_ws_codes_frame.va_addr);
        put_u64(&mut sample_c0_args, 24, self.tk_logits_out.va_addr);
        put_u32(&mut sample_c0_args, 32, self.talker_vocab as u32);

        let mut cp_args = [0u8; 16];
        put_u64(&mut cp_args, 0, self.cp_desc_gpu.va_addr);
        put_u64(&mut cp_args, 8, self.frame_ctrl.va_addr);

        let mut all_audio = Vec::<f32>::new();
        let mut pending_codes = Vec::<i32>::with_capacity(TTS_MAX_CODEC_CHUNK_FRAMES * 16);
        let mut stream_target_frames = 1usize; // Frame 0 flushes immediately (`80 ms` chunk) for <60ms TTFA, then ramps 1 -> 2 -> 4 -> 8!
        let mut emitted_frames = 0usize;
        let mut first_codes_ms = 0.0f64;
        let mut ttfa_ms = 0.0f64;
        let mut ar_decode_ms = 0.0f64;
        let mut codec_ms = 0.0f64;

        for step in 0..max_frames {
            if cancel.map_or(false, |c| c.load(Ordering::Relaxed)) {
                break;
            }

            let t_ar = Instant::now();
            unsafe {
                (*fctrl_ptr).frame_idx = step as i32;
                (*fctrl_ptr).is_eos = 0;
                (*fctrl_ptr).error_code = 0;
                if step > 0 {
                    (*tk_ctrl_ptr).start_pos = (t_ctx + step - 1) as i32;
                    (*tk_ctrl_ptr).error_code = 0;
                    (*tk_ctrl_ptr).steps_completed = 0;
                }
            }
            fence(Ordering::SeqCst);

            let mut frame_chain = Vec::with_capacity(3);
            if step > 0 {
                // Talker 28L decode step reading `tk_ext_embed` -> `tk_hidden_last` + `tk_logits_out`
                frame_chain.push(gpu.prepare_qmd(k_tk_mk, (num_sms, 1, 1), (512, 1, 1), &mk_tk_args, None));
            }
            frame_chain.push(gpu.prepare_qmd(k_sample_c0, (1, 1, 1), (256, 1, 1), &sample_c0_args, None));
            frame_chain.push(gpu.prepare_qmd(k_cp_mk, (num_sms, 1, 1), (512, 1, 1), &cp_args, None));

            Self::execute_qmd_chain(gpu, &frame_chain, 15)?;
            ar_decode_ms += t_ar.elapsed().as_secs_f64() * 1000.0;

            let (is_eos, tk_err, cp_err) = unsafe {
                (
                    std::ptr::read_volatile(&(*fctrl_ptr).is_eos),
                    std::ptr::read_volatile(&(*tk_ctrl_ptr).error_code),
                    std::ptr::read_volatile(&(*fctrl_ptr).error_code),
                )
            };
            if tk_err != 0 || cp_err != 0 {
                let _ = gpu.zero_vram(&self.tk_ws_bar_state);
                let _ = gpu.zero_vram(&self.cp_ws_bar_state);
                return Err(format!("TTS megakernel watchdog tripped (tk_err={}, cp_err={})", tk_err, cp_err));
            }
            if is_eos != 0 {
                break;
            }

            if step == 0 {
                first_codes_ms = t_total.elapsed().as_secs_f64() * 1000.0;
            }

            let frame_codes: [i32; 16] = unsafe {
                let base = (*fctrl_ptr).codes_out.as_ptr().add(step * 16);
                std::ptr::read_volatile(base as *const [i32; 16])
            };
            pending_codes.extend_from_slice(&frame_codes);
            emitted_frames += 1;

            if params.streaming_chunks && (pending_codes.len() / 16) >= stream_target_frames {
                let chunk_f = pending_codes.len() / 16;
                let t_cd = Instant::now();
                let wav_chunk = self.decode_codes_stream_chunk(gpu, module, &pending_codes, chunk_f)?;
                codec_ms += t_cd.elapsed().as_secs_f64() * 1000.0;
                pending_codes.clear();

                if ttfa_ms == 0.0 && !wav_chunk.is_empty() {
                    ttfa_ms = t_total.elapsed().as_secs_f64() * 1000.0;
                }
                all_audio.extend_from_slice(&wav_chunk);
                if !on_chunk(&wav_chunk) {
                    break;
                }
                if stream_target_frames < 8 {
                    stream_target_frames *= 2;
                }
            }
        }

        // Flush remaining buffered RVQ frames through `Code2Wav`
        if !pending_codes.is_empty() && !cancel.map_or(false, |c| c.load(Ordering::Relaxed)) {
            let chunk_f = pending_codes.len() / 16;
            let t_cd = Instant::now();
            let wav_chunk = self.decode_codes_stream_chunk(gpu, module, &pending_codes, chunk_f)?;
            codec_ms += t_cd.elapsed().as_secs_f64() * 1000.0;
            pending_codes.clear();

            if ttfa_ms == 0.0 && !wav_chunk.is_empty() {
                ttfa_ms = t_total.elapsed().as_secs_f64() * 1000.0;
            }
            all_audio.extend_from_slice(&wav_chunk);
            let _ = on_chunk(&wav_chunk);
        }

        let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
        if ttfa_ms == 0.0 {
            ttfa_ms = total_ms;
        }
        let n_samples = all_audio.len();
        let audio_sec = (n_samples as f64) / (TTS_SAMPLE_RATE as f64);
        let rtf = if audio_sec > 0.0 { (total_ms / 1000.0) / audio_sec } else { 0.0 };
        let speed_x_realtime = if rtf > 0.0 { 1.0 / rtf } else { 0.0 };
        let ar_ms_per_frame = if emitted_frames > 0 {
            ar_decode_ms / (emitted_frames as f64)
        } else {
            0.0
        };

        let stats = TtsStats {
            talker_model: self.model_name.clone(),
            speaker: eff_spk,
            language: eff_lang,
            prompt_ctx_rows: t_ctx,
            frames: emitted_frames,
            n_samples,
            audio_sec,
            prefill_ms,
            first_codes_ms,
            ttfa_ms,
            ar_decode_ms,
            ar_ms_per_frame,
            codec_ms,
            total_ms,
            rtf,
            speed_x_realtime,
        };

        Ok((all_audio, stats))
    }
}
