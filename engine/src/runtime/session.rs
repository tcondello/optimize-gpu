//! Unified RTX 3080 Hardware + Thinker (`Qwen` or `Inkling`) + Co-Resident Voice (`ASR` + `TTS`) Session.
//!
//! Owns `NvGpu` at the top level so `Qwen3-ASR` and `Qwen3-TTS` + `12Hz Code2Wav` co-reside in GDDR6X
//! alongside either `Qwen3-4B/8B` or `Inkling-2-Mini` (`0 ms` model swap, 100% RTX 3080 `sm_86`).

#![allow(dead_code)]

use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use super::json_escape;
use crate::gguf::Utf8StreamDecoder;
use crate::hw::{CubinModule, NvGpu, VramCheckpoint};
use crate::lora::LoraStatus;
use crate::models::inkling::{InklingEngine, InklingGenStats};
use crate::models::qwen::{QwenEngine, TopicRouteResult};
use crate::models::qwen_abi::{GenerationStats, EMBEDDED_CUBIN};
use crate::models::rag::{EmbedStrategy, RagDoc, RagSearchStats};
use crate::voice::asr::{AsrStats, QwenAsrEngine};
use crate::voice::tts::{QwenTtsEngine, TtsStats, TtsSynthParams};

enum ActiveThinker {
    Qwen(QwenEngine),
    Inkling(InklingEngine),
}

pub struct RuntimeSession {
    gpu: NvGpu,
    thinker: ActiveThinker,
    voice_cubin: Option<CubinModule>,
    voice_arena_cp: Option<VramCheckpoint>,
    asr: Option<QwenAsrEngine>,
    tts: Option<QwenTtsEngine>,
    max_ctx: usize,
}

pub fn format_stats_json(stats: &GenerationStats, token_ids: &[i32]) -> String {
    let ids_str: Vec<String> = token_ids.iter().map(|t| t.to_string()).collect();
    format!(
        "{{\"prompt_tokens\":{},\"cached_prefix_tokens\":{},\"prefill_new_tokens\":{},\"completion_tokens\":{},\
          \"prefill_ms\":{:.3},\"decode_ms\":{:.3},\"ttft_ms\":{:.3},\"total_ms\":{:.3},\
          \"decode_tok_per_s\":{:.2},\"prefill_tok_per_s\":{:.2},\"kernel_launches\":{},\
          \"spec_verify_calls\":{},\"spec_accepted_extra\":{},\"spec_crossover_pos\":{},\"spec_crossover_switched\":{},\
          \"stop_reason\":\"{}\",\"token_ids\":[{}]}}",
        stats.prompt_tokens,
        stats.cached_prefix_tokens,
        stats.prefill_new_tokens,
        stats.completion_tokens,
        stats.prefill_ms,
        stats.decode_ms,
        stats.ttft_ms,
        stats.total_ms,
        stats.decode_tok_per_s,
        stats.prefill_tok_per_s,
        stats.kernel_launches,
        stats.spec_verify_calls,
        stats.spec_accepted_extra,
        stats.spec_crossover_pos,
        stats.spec_crossover_switched,
        stats.stop_reason,
        ids_str.join(",")
    )
}

pub fn format_inkling_stats_json(st: &InklingGenStats) -> String {
    let ids_str: Vec<String> = st.token_ids.iter().map(|t| t.to_string()).collect();
    let prefill_new = st.prompt_tokens.saturating_sub(st.cached_prefix_tokens);
    format!(
        "{{\"prompt_tokens\":{},\"cached_prefix_tokens\":{},\"prefill_new_tokens\":{},\"completion_tokens\":{},\
          \"prefill_ms\":{:.3},\"decode_ms\":{:.3},\"ttft_ms\":{:.3},\"total_ms\":{:.3},\
          \"decode_tok_per_s\":{:.2},\"prefill_tok_per_s\":{:.2},\"kernel_launches\":{},\
          \"spec_verify_calls\":{},\"spec_accepted_extra\":{},\"spec_crossover_pos\":0,\"spec_crossover_switched\":false,\
          \"stop_reason\":\"{}\",\"token_ids\":[{}]}}",
        st.prompt_tokens,
        st.cached_prefix_tokens,
        prefill_new,
        st.completion_tokens,
        st.prefill_ms,
        st.decode_ms,
        st.ttft_ms,
        st.total_ms,
        st.decode_tok_per_s,
        st.prefill_tok_per_s,
        st.gpu_launches,
        st.spec_verify_calls,
        st.spec_accepted_extra,
        st.stop_reason,
        ids_str.join(",")
    )
}

pub fn format_lora_json(st: &LoraStatus) -> String {
    format!(
        "{{\"active\":{},\"path\":\"{}\",\"rank\":{},\"alpha\":{:.2},\"scale\":{:.4},\
          \"adapted_tensors\":{},\"vram_bytes\":{},\"swap_ms\":{:.3}}}",
        st.active,
        json_escape(&st.path),
        st.rank,
        st.alpha,
        st.scale,
        st.num_adapted_tensors,
        st.vram_bytes,
        st.swap_ms
    )
}

impl RuntimeSession {
    pub fn new(model_ref: &str, max_ctx: usize) -> Result<Self, String> {
        let mut gpu = NvGpu::init()?;
        let thinker = if InklingEngine::is_inkling_ref(model_ref) {
            ActiveThinker::Inkling(InklingEngine::load(&mut gpu, model_ref, max_ctx)?)
        } else {
            ActiveThinker::Qwen(QwenEngine::load(&mut gpu, model_ref, max_ctx)?)
        };
        Ok(Self {
            gpu,
            thinker,
            voice_cubin: None,
            voice_arena_cp: None,
            asr: None,
            tts: None,
            max_ctx,
        })
    }

    pub(super) fn warmup_for_daemon(&mut self) -> Result<(), String> {
        match self.thinker {
            ActiveThinker::Qwen(ref mut qwen) => {
                let warm_ids = qwen.encode_chat(&[("user".into(), "Hi".into())], false);
                let _ = qwen.generate(&mut self.gpu, &warm_ids, 2, 0.0, false, None, None, |_, _| true)?;
                qwen.reset_cache();
            }
            ActiveThinker::Inkling(ref mut ink) => {
                let warm_ids = ink.format_chat_prompt("Hi", None, Some("0"));
                let _ = ink.generate_with_cancel(
                    &mut self.gpu,
                    &warm_ids,
                    2,
                    0.0,
                    42,
                    None,
                    false,
                    true,
                    None,
                    false,
                    |_, _| true,
                )?;
                let _ = ink.reset_cache(&mut self.gpu);
            }
        }
        Ok(())
    }

    #[inline]
    pub(super) fn is_inkling(&self) -> bool {
        matches!(self.thinker, ActiveThinker::Inkling(_))
    }

    #[inline]
    pub(super) fn ping_info(&self) -> (&str, &str, u32) {
        match self.thinker {
            ActiveThinker::Qwen(ref q) => (q.model_ref(), q.model_name(), self.gpu.active_sms),
            ActiveThinker::Inkling(ref i) => (i.model_ref(), i.model_name(), self.gpu.active_sms),
        }
    }

    pub(super) fn telemetry_json(&self) -> String {
        let total_mb = (self.gpu.vram_size() as f64) / 1_048_576.0;
        let free_high_mb = (self.gpu.free_high_vram() as f64) / 1_048_576.0;
        let sys_mb = 480.0f64; // 224 MiB GSP-RM WPR2 + 256 MiB BAR1 ring/PT window
        let used_high_mb = (total_mb - sys_mb - free_high_mb).max(0.0);

        let asr_mb = self
            .asr
            .as_ref()
            .map(|a| (a.vram_bytes_total as f64) / 1_048_576.0)
            .unwrap_or(0.0);
        let asr_name = self.asr.as_ref().map(|a| a.model_name()).unwrap_or("");
        let tts_mb = self.tts.as_ref().map(|t| t.total_vram_mb).unwrap_or(0.0);
        let tts_name = self.tts.as_ref().map(|t| t.model_name()).unwrap_or("");

        let (aux_mb, aux_label) = match self.thinker {
            ActiveThinker::Inkling(ref ink) => {
                if ink.has_dspark() {
                    (447.3, "DSpark MTP")
                } else {
                    (0.0, "reclaimed")
                }
            }
            ActiveThinker::Qwen(ref q) => {
                let dj = q.drafter_status_json(None);
                let lj = q.lora_status_json();
                let dr_mb = if dj.contains("\"active\":true") { 640.0 } else { 0.0 };
                let lora_b = crate::topic_lora::json_find_num(&lj, "vram_bytes").unwrap_or(0.0);
                let l_mb = lora_b / 1_048_576.0;
                let lbl = if dr_mb > 0.0 && l_mb > 1.0 {
                    "Drafter+LoRA"
                } else if dr_mb > 0.0 {
                    "0.6B Drafter"
                } else if l_mb > 1.0 {
                    "LoRA Arena"
                } else {
                    "standby"
                };
                (dr_mb + l_mb, lbl)
            }
        };
        let voice_mb = asr_mb + tts_mb;
        let thinker_mb = (used_high_mb - voice_mb - aux_mb).max(0.0);

        format!(
            "{{\"total_mb\":{:.1},\"thinker_mb\":{:.1},\"aux_mb\":{:.1},\"aux_label\":\"{}\",\
              \"voice_mb\":{:.1},\"asr_mb\":{:.1},\"asr_name\":\"{}\",\"tts_mb\":{:.1},\"tts_name\":\"{}\",\
              \"free_mb\":{:.1},\"sys_mb\":{:.1},\"max_ctx\":{},\"active_sms\":{}}}",
            total_mb,
            thinker_mb,
            aux_mb,
            json_escape(aux_label),
            voice_mb,
            asr_mb,
            json_escape(asr_name),
            tts_mb,
            json_escape(tts_name),
            free_high_mb,
            sys_mb,
            self.max_ctx,
            self.gpu.active_sms
        )
    }

    #[inline]
    pub(super) fn model_name(&self) -> &str {
        self.ping_info().1
    }

    pub(super) fn reset_cache(&mut self) {
        match self.thinker {
            ActiveThinker::Qwen(ref mut q) => q.reset_cache(),
            ActiveThinker::Inkling(ref mut i) => {
                let _ = i.reset_cache(&mut self.gpu);
            }
        }
    }

    pub(super) fn keepalive_or_recover(&mut self) {
        if let Err(e) = self.gpu.keepalive_compute_tick() {
            let mref = self.ping_info().0.to_string();
            let ctx = self.max_ctx;
            eprintln!("[rtx daemon] Compute keepalive error ({}); re-initializing GPU...", e);
            self.gpu.shutdown_clean();
            if let Ok(fresh) = Self::new(&mref, ctx) {
                *self = fresh;
            }
        }
    }

    pub fn shutdown_clean(&mut self) {
        self.gpu.shutdown_clean();
    }

    // -------------------------------------------------------------------------
    // Shared 100%-on-RTX-3080 Voice Stack (`Qwen3-ASR` + `Qwen3-TTS` + `Code2Wav`)
    // -------------------------------------------------------------------------

    fn resolve_voice_ref(&mut self, raw_ref: &str, kind: &str) -> String {
        let trimmed = raw_ref.trim();
        if let ActiveThinker::Inkling(ref mut ink) = self.thinker {
            let reclaimed = ink.reclaim_dspark_for_voice(&mut self.gpu);
            if reclaimed > 0.0 {
                eprintln!(
                    "[rtx] Reclaimed {:.1} MiB DSpark VRAM so co-resident 0.6B STT + TTS fit on 10 GB RTX 3080",
                    reclaimed
                );
            }
            if matches!(trimmed, "" | "default" | "1.7b" | "1.7b-q8" | "q8_0") {
                eprintln!(
                    "[rtx] Auto-selecting 0.6B {} checkpoint alongside 7.5 GiB Inkling-2-Mini (free high VRAM: {:.1} MiB)",
                    kind,
                    (self.gpu.free_high_vram() as f64) / (1024.0 * 1024.0)
                );
                return "0.6b".to_string();
            }
        }
        if trimmed.is_empty() {
            "1.7b".to_string()
        } else {
            trimmed.to_string()
        }
    }

    fn ensure_voice_cubin(&mut self) -> Result<(), String> {
        if self.voice_cubin.is_none() {
            self.voice_cubin = Some(self.gpu.load_cubin(EMBEDDED_CUBIN)?);
            self.voice_arena_cp = Some(self.gpu.checkpoint_vram());
        }
        Ok(())
    }

    pub(super) fn warm_asr(&mut self, asr_ref: &str) -> Result<(String, f64), String> {
        let eff_ref = self.resolve_voice_ref(asr_ref, "Qwen3-ASR");
        self.ensure_voice_cubin()?;
        let need_load = self.asr.as_ref().map_or(true, |a| a.asr_ref() != eff_ref);
        if need_load {
            let need_bytes = if eff_ref.starts_with("0.6b") {
                1050u64 << 20
            } else {
                2350u64 << 20
            };
            if self.gpu.free_high_vram() < need_bytes {
                if let Some(cp) = self.voice_arena_cp {
                    self.asr = None;
                    self.tts = None;
                    self.gpu.restore_vram_checkpoint(cp);
                }
            }
            let eng = QwenAsrEngine::load(&mut self.gpu, &eff_ref)?;
            self.asr = Some(eng);
        }
        let asr = self.asr.as_ref().unwrap();
        Ok((
            asr.model_name().to_string(),
            (asr.vram_bytes_total as f64) / (1024.0 * 1024.0),
        ))
    }

    pub fn transcribe_audio(
        &mut self,
        asr_ref: &str,
        pcm_16k: &[f32],
        language_hint: Option<&str>,
    ) -> Result<AsrStats, String> {
        let _ = self.warm_asr(asr_ref)?;
        let asr = self.asr.as_mut().unwrap();
        let cubin = self.voice_cubin.as_ref().unwrap();
        asr.transcribe(&mut self.gpu, cubin, pcm_16k, language_hint)
    }

    pub(super) fn warm_tts(&mut self, talker_ref: &str) -> Result<(String, f64, String), String> {
        let eff_ref = self.resolve_voice_ref(talker_ref, "Qwen3-TTS");
        self.ensure_voice_cubin()?;
        let need_load = self.tts.as_ref().map_or(true, |t| t.talker_ref() != eff_ref);
        if need_load {
            let need_bytes = if eff_ref.starts_with("0.6b") {
                1610u64 << 20
            } else {
                2700u64 << 20
            };
            if self.gpu.free_high_vram() < need_bytes {
                if let Some(cp) = self.voice_arena_cp {
                    self.asr = None;
                    self.tts = None;
                    self.gpu.restore_vram_checkpoint(cp);
                }
            }
            let eng = QwenTtsEngine::load(&mut self.gpu, &eff_ref, None)?;
            self.tts = Some(eng);
        }
        let tts = self.tts.as_ref().unwrap();
        Ok((
            tts.model_name().to_string(),
            tts.total_vram_mb,
            tts.speaker_names().join(","),
        ))
    }

    pub fn synthesize_speech<F>(
        &mut self,
        talker_ref: &str,
        params: &TtsSynthParams,
        cancel: Option<&AtomicBool>,
        on_chunk: F,
    ) -> Result<(Vec<f32>, TtsStats), String>
    where
        F: FnMut(&[f32]) -> bool,
    {
        let _ = self.warm_tts(talker_ref)?;
        let tts = self.tts.as_mut().unwrap();
        let cubin = self.voice_cubin.as_ref().unwrap();
        tts.synthesize(&mut self.gpu, cubin, params, cancel, on_chunk)
    }

    pub(super) fn decode_codec_frames(
        &mut self,
        talker_ref: &str,
        codes: &[i32],
        n_frames: usize,
    ) -> Result<(Vec<f32>, f64), String> {
        let _ = self.warm_tts(talker_ref)?;
        let tts = self.tts.as_mut().unwrap();
        tts.reset_codec_stream(&mut self.gpu)?;
        let cubin = self.voice_cubin.as_ref().unwrap();
        let t0 = Instant::now();
        let wav = tts.decode_codes_stream_chunk(&mut self.gpu, cubin, codes, n_frames)?;
        Ok((wav, t0.elapsed().as_secs_f64() * 1000.0))
    }

    // -------------------------------------------------------------------------
    // Unified Text Generation & Thinker-Specific Operations
    // -------------------------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    pub(super) fn stream_generate<F>(
        &mut self,
        mode: &str,
        payload: &[u8],
        max_tokens: usize,
        temp: f32,
        spec: bool,
        reset: bool,
        extra_opt: Option<&str>,
        cancel: Option<&AtomicBool>,
        mut on_piece: F,
    ) -> Result<String, String>
    where
        F: FnMut(i32, &str) -> bool,
    {
        match self.thinker {
            ActiveThinker::Inkling(ref mut ink) => {
                if let Some(g_s) = extra_opt {
                    if let Ok(g_v) = g_s.parse::<usize>() {
                        if (1..=7).contains(&g_v) {
                            ink.set_dspark_gamma(g_v);
                        }
                    }
                }
                let prompt_ids: Vec<i32> = if mode == "ids" {
                    payload
                        .chunks_exact(4)
                        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                        .collect()
                } else if mode == "raw" {
                    ink.encode_raw(&String::from_utf8_lossy(payload))
                } else if mode == "chat_sys" && payload.len() >= 4 {
                    let sys_len = u32::from_le_bytes(payload[0..4].try_into().unwrap()) as usize;
                    let sys_end = (4 + sys_len).min(payload.len());
                    let sys_s = String::from_utf8_lossy(&payload[4..sys_end]).into_owned();
                    let usr_s = String::from_utf8_lossy(&payload[sys_end..]).into_owned();
                    ink.format_chat_prompt(
                        &usr_s,
                        if sys_s.is_empty() { None } else { Some(sys_s.as_str()) },
                        None,
                    )
                } else {
                    ink.format_chat_prompt(&String::from_utf8_lossy(payload), None, None)
                };

                let st = ink.generate_with_cancel(
                    &mut self.gpu,
                    &prompt_ids,
                    max_tokens,
                    temp,
                    42,
                    None,
                    spec,
                    reset,
                    cancel,
                    mode == "raw",
                    on_piece,
                )?;
                Ok(format_inkling_stats_json(&st))
            }
            ActiveThinker::Qwen(ref mut qwen) => {
                let crossover_override = extra_opt.and_then(|co_s| co_s.parse::<usize>().ok());
                if reset {
                    qwen.reset_cache();
                }
                let prompt_ids: Vec<i32> = if mode == "ids" {
                    payload
                        .chunks_exact(4)
                        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                        .collect()
                } else if mode == "raw" {
                    qwen.encode_raw_prompt(&String::from_utf8_lossy(payload))
                } else if mode == "chat_sys" && payload.len() >= 4 {
                    let sys_len = u32::from_le_bytes(payload[0..4].try_into().unwrap()) as usize;
                    let sys_end = (4 + sys_len).min(payload.len());
                    let sys_s = String::from_utf8_lossy(&payload[4..sys_end]).into_owned();
                    let usr_s = String::from_utf8_lossy(&payload[sys_end..]).into_owned();
                    let mut msgs = Vec::with_capacity(2);
                    if !sys_s.is_empty() {
                        msgs.push(("system".to_string(), sys_s));
                    }
                    msgs.push(("user".to_string(), usr_s));
                    qwen.encode_chat(&msgs, false)
                } else {
                    let s = String::from_utf8_lossy(payload).into_owned();
                    qwen.encode_chat(&[("user".to_string(), s)], true)
                };

                let mut utf8_dec = Utf8StreamDecoder::new();
                let mut alive = true;
                let (out_ids, stats) = qwen.generate(
                    &mut self.gpu,
                    &prompt_ids,
                    max_tokens,
                    temp,
                    spec,
                    crossover_override,
                    cancel,
                    |tok_id, tok_bytes| {
                        let piece = utf8_dec.push(tok_bytes);
                        if !on_piece(tok_id, &piece) {
                            alive = false;
                            return false;
                        }
                        true
                    },
                )?;
                if alive {
                    let tail = utf8_dec.flush();
                    if !tail.is_empty() {
                        let _ = on_piece(-1, &tail);
                    }
                }
                Ok(format_stats_json(&stats, &out_ids))
            }
        }
    }

    pub(super) fn http_chat_completion<F>(
        &mut self,
        msgs: &[(String, String)],
        max_toks: usize,
        temp: f32,
        spec: bool,
        cancel: Option<&AtomicBool>,
        mut on_piece: F,
    ) -> Result<(String, &'static str, usize, usize), String>
    where
        F: FnMut(&str) -> bool,
    {
        let mut full_text = String::new();
        match self.thinker {
            ActiveThinker::Inkling(ref mut ink) => {
                let prompt_ids = ink.format_multiturn_chat(msgs, Some("0"));
                let st = ink.generate_with_cancel(
                    &mut self.gpu,
                    &prompt_ids,
                    max_toks,
                    temp,
                    42,
                    None,
                    spec,
                    false,
                    cancel,
                    false,
                    |_, piece| {
                        full_text.push_str(piece);
                        on_piece(piece)
                    },
                )?;
                Ok((full_text, st.stop_reason, st.prompt_tokens, st.completion_tokens))
            }
            ActiveThinker::Qwen(ref mut qwen) => {
                let prompt_ids = qwen.encode_chat(msgs, true);
                let mut dec = Utf8StreamDecoder::new();
                let (_, st) = qwen.generate(
                    &mut self.gpu,
                    &prompt_ids,
                    max_toks,
                    temp,
                    spec,
                    None,
                    cancel,
                    |_, tbytes| {
                        let piece = dec.push(tbytes);
                        if !piece.is_empty() {
                            full_text.push_str(&piece);
                            if !on_piece(&piece) {
                                return false;
                            }
                        }
                        true
                    },
                )?;
                let tail = dec.flush();
                if !tail.is_empty() {
                    full_text.push_str(&tail);
                    let _ = on_piece(&tail);
                }
                Ok((full_text, st.stop_reason, st.prompt_tokens, st.completion_tokens))
            }
        }
    }

    pub(super) fn lora_status_json(&self) -> String {
        match self.thinker {
            ActiveThinker::Qwen(ref q) => q.lora_status_json(),
            ActiveThinker::Inkling(_) => {
                "{\"active\":false,\"path\":\"\",\"rank\":0,\"alpha\":0.00,\"scale\":0.0000,\"adapted_tensors\":0,\"vram_bytes\":0,\"swap_ms\":0.000}".to_string()
            }
        }
    }

    pub(super) fn lora_clear_json(&mut self) -> Result<String, String> {
        match self.thinker {
            ActiveThinker::Qwen(ref mut q) => Ok(format_lora_json(&q.clear_lora(&mut self.gpu)?)),
            ActiveThinker::Inkling(_) => Ok(self.lora_status_json()),
        }
    }

    pub(super) fn lora_load_json(&mut self, path: &str, scale: f32) -> Result<String, String> {
        match self.thinker {
            ActiveThinker::Qwen(ref mut q) => Ok(format_lora_json(&q.load_lora(&mut self.gpu, path, scale)?)),
            ActiveThinker::Inkling(_) => Err("PEFT LoRA hot-swap requires Qwen3 dense backbone".into()),
        }
    }

    pub(super) fn rag_status_json(&self) -> String {
        match self.thinker {
            ActiveThinker::Qwen(ref q) => q.rag_status_json(None),
            ActiveThinker::Inkling(_) => "{\"active\":false,\"num_docs\":0}".to_string(),
        }
    }

    pub(super) fn rag_clear(&mut self) {
        if let ActiveThinker::Qwen(ref mut q) = self.thinker {
            q.clear_rag_qvec();
        }
    }

    pub(super) fn rag_load_json(&mut self, path: &Path) -> Result<String, String> {
        match self.thinker {
            ActiveThinker::Qwen(ref mut q) => {
                let t0 = Instant::now();
                q.load_rag_qvec(&mut self.gpu, path)?;
                Ok(q.rag_status_json(Some(t0.elapsed().as_secs_f64() * 1000.0)))
            }
            ActiveThinker::Inkling(_) => Err("Same-model .qvec RAG requires Qwen3 backbone".into()),
        }
    }

    pub(super) fn rag_index(
        &mut self,
        docs: &[RagDoc],
        strat: EmbedStrategy,
        lora_path: &str,
        out_path: &Path,
        load_into_vram: bool,
    ) -> Result<(f64, f64), String> {
        match self.thinker {
            ActiveThinker::Qwen(ref mut q) => {
                q.build_qvec_index(&mut self.gpu, docs, strat, lora_path, out_path, load_into_vram)
            }
            ActiveThinker::Inkling(_) => Err("Same-model .qvec indexing requires Qwen3 backbone".into()),
        }
    }

    pub(super) fn rag_search(&mut self, query: &str, top_k: usize) -> Result<RagSearchStats, String> {
        match self.thinker {
            ActiveThinker::Qwen(ref mut q) => q.rag_search(&mut self.gpu, query, top_k),
            ActiveThinker::Inkling(_) => Err("Same-model .qvec search requires Qwen3 backbone".into()),
        }
    }

    pub(super) fn embed_text(
        &mut self,
        strat: EmbedStrategy,
        text: &str,
        is_query: bool,
        lora_override: Option<&str>,
    ) -> Result<(Vec<f32>, f64), String> {
        match self.thinker {
            ActiveThinker::Qwen(ref mut q) => q.embed_text(&mut self.gpu, strat, text, is_query, lora_override),
            ActiveThinker::Inkling(_) => Err("Same-model embeddings require Qwen3 backbone".into()),
        }
    }

    pub(super) fn drafter_status_json(&self) -> String {
        match self.thinker {
            ActiveThinker::Qwen(ref q) => q.drafter_status_json(None),
            ActiveThinker::Inkling(ref i) => i.dspark_status_json(),
        }
    }

    pub(super) fn drafter_clear(&mut self) {
        if let ActiveThinker::Qwen(ref mut q) = self.thinker {
            q.clear_drafter();
        }
    }

    pub(super) fn drafter_load_json(&mut self, draft_ref: &str, prior_p: &str) -> Result<String, String> {
        match self.thinker {
            ActiveThinker::Qwen(ref mut q) => {
                let t0 = Instant::now();
                q.ensure_drafter_loaded(&mut self.gpu, draft_ref)?;
                if !prior_p.is_empty() {
                    let _ = q.load_drafter_prior(&mut self.gpu, Path::new(prior_p))?;
                }
                Ok(q.drafter_status_json(Some(t0.elapsed().as_secs_f64() * 1000.0)))
            }
            ActiveThinker::Inkling(ref i) => Ok(i.dspark_status_json()),
        }
    }

    pub(super) fn inkling_set_gamma_json(&mut self, gamma_opt: Option<usize>) -> String {
        match self.thinker {
            ActiveThinker::Inkling(ref mut ink) => {
                if let Some(g) = gamma_opt {
                    ink.set_dspark_gamma(g);
                }
                format!("{{\"gamma\":{},\"dspark_loaded\":{}}}", ink.dspark_gamma(), ink.has_dspark())
            }
            ActiveThinker::Qwen(_) => "{\"gamma\":0,\"dspark_loaded\":false}".to_string(),
        }
    }

    pub(super) fn topic_load_json(&mut self, bank_dir: &Path) -> Result<String, String> {
        match self.thinker {
            ActiveThinker::Qwen(ref mut q) => {
                let t0 = Instant::now();
                let (canonical, n_topics, n_docs) = q.load_topic_bank(&mut self.gpu, bank_dir)?;
                let (used_slots, max_slots, embed_pinned) = q.lora_cache_summary();
                Ok(format!(
                    "{{\"active\":true,\"bank_dir\":\"{}\",\"num_topics\":{},\"num_docs\":{},\"cached_topic_slots\":{},\"max_topic_slots\":{},\"embed_lora_pinned\":{},\"load_ms\":{:.2}}}",
                    json_escape(&canonical),
                    n_topics,
                    n_docs,
                    used_slots,
                    max_slots,
                    embed_pinned,
                    t0.elapsed().as_secs_f64() * 1000.0
                ))
            }
            ActiveThinker::Inkling(_) => Err("Topic LoRA bank requires Qwen3 backbone".into()),
        }
    }

    pub(super) fn topic_status_json(&self) -> String {
        match self.thinker {
            ActiveThinker::Qwen(ref q) => q.topic_status_json(&q.lora_status_json()),
            ActiveThinker::Inkling(_) => "{\"active\":false,\"num_topics\":0}".to_string(),
        }
    }

    pub(super) fn topic_list_json(&self) -> String {
        match self.thinker {
            ActiveThinker::Qwen(ref q) => q.topic_list_json(),
            ActiveThinker::Inkling(_) => "[]".to_string(),
        }
    }

    pub(super) fn topic_clear(&mut self) {
        if let ActiveThinker::Qwen(ref mut q) = self.thinker {
            let _ = q.clear_topic_bank(&mut self.gpu);
        }
    }

    pub(super) fn topic_route(
        &mut self,
        query: &str,
        top_k: usize,
        min_score_opt: Option<f32>,
        auto_load: bool,
    ) -> Result<(TopicRouteResult, (usize, usize, bool)), String> {
        match self.thinker {
            ActiveThinker::Qwen(ref mut q) => {
                let rt = q.topic_route(&mut self.gpu, query, top_k, min_score_opt, auto_load)?;
                Ok((rt, q.lora_cache_summary()))
            }
            ActiveThinker::Inkling(_) => Err("Topic LoRA routing requires Qwen3 backbone".into()),
        }
    }

    pub(super) fn extract_residual_hooks(&mut self, toks: &[i32]) -> Result<Vec<f32>, String> {
        match self.thinker {
            ActiveThinker::Qwen(ref mut q) => q.extract_residual_hooks(&mut self.gpu, toks),
            ActiveThinker::Inkling(_) => Err("Residual hooks require Qwen3 backbone".into()),
        }
    }

    pub(super) fn eval_next_token_logits(&mut self, toks: &[i32]) -> Result<Vec<f32>, String> {
        match self.thinker {
            ActiveThinker::Qwen(ref mut q) => q.eval_next_token_logits(&mut self.gpu, toks),
            ActiveThinker::Inkling(_) => Err("Logits eval requires Qwen3 backbone".into()),
        }
    }

    pub(super) fn extract_mlp_kv_hooks(
        &mut self,
        toks: &[i32],
        layers: &[usize],
        pos_start: usize,
        pos_count: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        match self.thinker {
            ActiveThinker::Qwen(ref mut q) => {
                q.extract_mlp_kv_hooks(&mut self.gpu, toks, layers, pos_start, pos_count)
            }
            ActiveThinker::Inkling(_) => Err("MLP KV hooks require Qwen3 backbone".into()),
        }
    }
}
