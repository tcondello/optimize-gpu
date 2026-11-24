//! Native macOS ARM64 Informative 1990s Demoscene × Charm Terminal UI (`./tui` / `rtx tui`).

mod composer;
mod demoscene;
mod markdown;
mod telemetry;
mod terminal;

use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use composer::Composer;
use demoscene::{copper_gradient, filter_slash_commands, DemosceneEngine, TelemetrySnapshot};
use markdown::{
    extract_last_fenced_code, format_assistant_markdown, format_help_markdown,
    format_markdown_block, format_verify_markdown, sanitize_text_for_speech, select_bench_prompt,
    split_spoken_sentences, RenderedLine, Span,
};
use telemetry::{
    measure_voice_self_test_tts, prepare_voice_self_test_pcm, query_hw_telemetry_from_daemon,
    query_lora_badge_from_daemon, query_rag_badge_from_daemon, query_topic_badge_from_daemon,
    spawn_generation_stream, spawn_tts_sentences, HwTelemetry, StreamEvent, TurnStats,
};
#[cfg(test)]
use telemetry::parse_stats_json;
use terminal::{
    copy_to_macos_clipboard, poll_input_events, term_size, InputEvent, RawTermGuard, ScreenBuf,
    Style, BG_BADGE_ON, BG_BADGE_T1, FG_AMBER, FG_CORAL, FG_CYAN, FG_GREEN, FG_PURPLE,
};

use crate::gguf::{resolve_model_path, BpeTokenizer, GgufFile, ModelConfig};
use crate::models::inkling::{InklingEngine, TOK_CONTENT_TEXT, TOK_END_MESSAGE, TOK_MSG_MODEL};
use crate::models::rag::format_rag_augmented_prompt;
use crate::runtime::client::{DaemonControl, DaemonQuery, DaemonRagClient, DaemonStream};
use crate::voice::audio::{read_wav_mono_16k, CoreAudioRecorder, ASR_SAMPLE_RATE, TTS_SAMPLE_RATE};

#[derive(Clone)]
struct ChatMsg {
    role: String,
    content: String,
    raw_token_ids: Vec<i32>,
    stats: Option<TurnStats>,
}

struct TuiApp {
    #[allow(dead_code)]
    model_ref: String,
    gguf_path: PathBuf,
    gguf_size_gb: f64,
    active_gb_per_tok: f64,
    is_inkling: bool,
    inkling_effort: String,
    inkling_gamma: usize,
    tok: BpeTokenizer,
    cfg: ModelConfig,
    active_sms: u32,
    ping_ms: f64,
    daemon_online: bool,
    spec_enabled: bool,
    temperature: f32,
    max_tokens: usize,
    system_prompt: String,
    voice_enabled: bool,
    voice_speaker: String,
    voice_lang: String,
    tts_model_ref: String,
    tts_speaking: Arc<AtomicBool>,
    tts_cancel_flag: Option<Arc<AtomicBool>>,
    last_ttfa_ms: Arc<std::sync::Mutex<f64>>,
    lora_badge: Option<String>,
    rag_badge: Option<String>,
    rag_top_k: usize,
    topic_badge: Option<String>,
    topic_lora_active: bool,
    asr_model_ref: String,
    asr_lang_hint: String,
    active_recorder: Option<CoreAudioRecorder>,
    asr_rx: Option<Receiver<Result<(String, String, f64), String>>>,
    last_asr_ms: f64,
    hands_free_talk: bool,
    prev_tts_speaking: bool,
    talk_rearm_at: Option<Instant>,
    messages: Vec<ChatMsg>,
    composer: Composer,
    scroll_from_bottom: usize,
    total_ctx_used: usize,
    last_cached_prefix: usize,
    last_tps: f64,
    last_ttft_ms: f64,
    last_prefill_tps: f64,
    last_launches: usize,
    last_spec_extra: usize,
    last_think_words: usize,
    last_answer_words: usize,
    expand_thinking: bool,
    slash_sel: usize,
    hw_telemetry: Option<HwTelemetry>,
    streaming_rx: Option<Receiver<StreamEvent>>,
    cancel_flag: Option<Arc<AtomicBool>>,
    stream_live_tps: f64,
    stream_live_toks: usize,
    spinner_tick: usize,
    toast: Option<(String, Instant)>,
    mouse_capture: bool,
    demo: DemosceneEngine,
    viz_pcm: Arc<Mutex<Vec<f32>>>,
    mic_viz_scratch: Vec<f32>,
}

impl TuiApp {
    fn new(
        model_ref: &str,
        max_ctx: usize,
        spec: bool,
        temp: f32,
        max_tokens: usize,
    ) -> Result<Self, String> {
        DaemonControl::ensure_running(model_ref, max_ctx)?;
        let t_ping = Instant::now();
        let (_, _, active_sms) = DaemonControl::ping()?;
        let ping_ms = t_ping.elapsed().as_secs_f64() * 1000.0;

        let gguf_path = resolve_model_path(model_ref)?;
        let gguf_bytes = std::fs::metadata(&gguf_path).map(|m| m.len()).unwrap_or(0);
        let gguf_size_gb = gguf_bytes as f64 / 1e9;
        let gguf = GgufFile::open(&gguf_path)?;
        let tok = gguf.build_tokenizer()?;
        let cfg = ModelConfig::from_gguf(&gguf, &tok, max_ctx, model_ref)?;
        let is_inkling = cfg.arch == "inkling2_loop";
        let active_gb_per_tok = if is_inkling { 2.68 } else { gguf_size_gb };
        let init_effort = std::env::var("RTX_EFFORT")
            .or_else(|_| std::env::var("INKLING_EFFORT"))
            .ok()
            .map(|s| InklingEngine::normalize_effort(&s).to_string())
            .unwrap_or_else(|| "0.2".to_string());
        let init_gamma = std::env::var("RTX_GAMMA")
            .or_else(|_| std::env::var("INKLING_GAMMA"))
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(7)
            .clamp(1, 7);
        if is_inkling {
            let _ = DaemonControl::query(DaemonQuery::InklingSetGamma(init_gamma));
        }

        let lora_badge = if is_inkling { None } else { query_lora_badge_from_daemon() };
        let topic_badge = if is_inkling { None } else { query_topic_badge_from_daemon() };
        let rag_badge = if is_inkling || topic_badge.is_some() {
            None
        } else {
            query_rag_badge_from_daemon()
        };
        let hw_telemetry = query_hw_telemetry_from_daemon();

        Ok(Self {
            model_ref: model_ref.to_string(),
            gguf_path,
            gguf_size_gb,
            active_gb_per_tok,
            is_inkling,
            inkling_effort: init_effort,
            inkling_gamma: init_gamma,
            tok,
            cfg,
            active_sms,
            ping_ms,
            daemon_online: true,
            spec_enabled: spec,
            temperature: temp,
            max_tokens: max_tokens.max(64),
            system_prompt: String::new(),
            voice_enabled: false,
            voice_speaker: "ryan".to_string(),
            voice_lang: "english".to_string(),
            tts_model_ref: (if is_inkling { "0.6b" } else { "1.7b" }).to_string(),
            tts_speaking: Arc::new(AtomicBool::new(false)),
            tts_cancel_flag: None,
            last_ttfa_ms: Arc::new(std::sync::Mutex::new(0.0)),
            lora_badge,
            rag_badge,
            rag_top_k: 3,
            topic_badge,
            topic_lora_active: false,
            asr_model_ref: (if is_inkling { "0.6b" } else { "1.7b" }).to_string(),
            asr_lang_hint: "auto".to_string(),
            active_recorder: None,
            asr_rx: None,
            last_asr_ms: 0.0,
            hands_free_talk: false,
            prev_tts_speaking: false,
            talk_rearm_at: None,
            messages: Vec::new(),
            composer: Composer::new(),
            scroll_from_bottom: 0,
            total_ctx_used: 0,
            last_cached_prefix: 0,
            last_tps: 0.0,
            last_ttft_ms: 0.0,
            last_prefill_tps: 0.0,
            last_launches: 0,
            last_spec_extra: 0,
            last_think_words: 0,
            last_answer_words: 0,
            expand_thinking: false,
            slash_sel: 0,
            hw_telemetry,
            streaming_rx: None,
            cancel_flag: None,
            stream_live_tps: 0.0,
            stream_live_toks: 0,
            spinner_tick: 0,
            toast: None,
            mouse_capture: true,
            demo: DemosceneEngine::new(),
            viz_pcm: Arc::new(Mutex::new(Vec::with_capacity(1024))),
            mic_viz_scratch: Vec::with_capacity(512),
        })
    }

    fn snapshot(&self) -> TelemetrySnapshot<'_> {
        let active_tps = if self.streaming_rx.is_some() && self.stream_live_tps > 0.0 {
            self.stream_live_tps
        } else {
            self.last_tps
        };
        let last_ttfa_ms = self.last_ttfa_ms.lock().map(|g| *g).unwrap_or(0.0);
        TelemetrySnapshot {
            model_name: &self.cfg.model_name,
            is_inkling: self.is_inkling,
            gguf_size_gb: self.gguf_size_gb,
            active_gb_per_tok: self.active_gb_per_tok,
            active_sms: self.active_sms,
            ping_ms: self.ping_ms,
            daemon_online: self.daemon_online,
            effort: &self.inkling_effort,
            spec_on: self.spec_enabled,
            gamma: self.inkling_gamma,
            temp: self.temperature,
            max_tokens: self.max_tokens,
            ctx_used: self.total_ctx_used,
            max_ctx: self.cfg.max_ctx,
            cached_prefix: self.last_cached_prefix,
            active_tps,
            last_ttft_ms: self.last_ttft_ms,
            last_prefill_tps: self.last_prefill_tps,
            last_launches: self.last_launches,
            last_spec_extra: self.last_spec_extra,
            think_words: self.last_think_words,
            answer_words: self.last_answer_words,
            expand_thinking: self.expand_thinking,
            voice_enabled: self.voice_enabled,
            hands_free: self.hands_free_talk,
            voice_speaker: &self.voice_speaker,
            voice_model_ref: &self.asr_model_ref,
            last_asr_ms: self.last_asr_ms,
            last_ttfa_ms,
            is_recording: self.active_recorder.is_some(),
            is_transcribing: self.asr_rx.is_some(),
            is_speaking: self.tts_speaking.load(Ordering::Relaxed),
            is_streaming: self.streaming_rx.is_some(),
            hw: self.hw_telemetry.as_ref(),
        }
    }

    fn set_toast(&mut self, msg: impl Into<String>) {
        self.toast = Some((msg.into(), Instant::now()));
    }

    fn push_info(&mut self, content: impl Into<String>) {
        self.messages.push(ChatMsg {
            role: "info".into(),
            content: content.into(),
            raw_token_ids: Vec::new(),
            stats: None,
        });
        self.scroll_from_bottom = 0;
    }

    fn stop_active_tts(&mut self) {
        if let Some(c) = &self.tts_cancel_flag {
            c.store(true, Ordering::Relaxed);
        }
        self.tts_speaking.store(false, Ordering::Relaxed);
    }

    fn spawn_tts_for_text(&mut self, raw_markdown: &str) {
        let clean = sanitize_text_for_speech(raw_markdown);
        if clean.is_empty() {
            return;
        }
        let _ = self.demo.configure("mod", "off", 120, &self.viz_pcm);
        self.stop_active_tts();
        let sentences = split_spoken_sentences(&clean);
        if sentences.is_empty() {
            return;
        }
        let cancel = spawn_tts_sentences(
            sentences,
            self.tts_model_ref.clone(),
            self.voice_speaker.clone(),
            self.voice_lang.clone(),
            Arc::clone(&self.tts_speaking),
            Arc::clone(&self.last_ttfa_ms),
            Arc::clone(&self.viz_pcm),
        );
        self.tts_cancel_flag = Some(cancel);
    }

    fn encode_conversation(&self, last_user_override: Option<&str>, sys_override: Option<&str>) -> Vec<i32> {
        let eff_sys = if !self.system_prompt.is_empty() {
            Some(self.system_prompt.as_str())
        } else {
            sys_override.filter(|s| !s.is_empty())
        };
        let last_user_idx = self.messages.iter().rposition(|m| m.role == "user");
        let last_user_txt = last_user_override
            .or_else(|| last_user_idx.map(|i| self.messages[i].content.as_str()))
            .unwrap_or("");
        let user_no_think = last_user_txt.contains("/no_think") || self.voice_enabled || self.topic_lora_active;

        if self.is_inkling {
            let mut ids = Vec::with_capacity(1024);
            if let Some(sys_s) = eff_sys {
                if !sys_s.trim().is_empty() {
                    let mut sh = String::from("<|message_system|><|content_text|>");
                    sh.push_str(sys_s.trim());
                    sh.push_str("<|end_message|>");
                    ids.extend(self.tok.encode(&sh));
                }
            }
            let eff_effort = if user_no_think { "0" } else { self.inkling_effort.as_str() };
            let mut eff_hdr = String::from("<|message_system|><|content_text|>Thinking effort level: ");
            eff_hdr.push_str(eff_effort);
            eff_hdr.push_str("<|end_message|>");
            ids.extend(self.tok.encode(&eff_hdr));

            for (idx, m) in self.messages.iter().enumerate() {
                if m.role == "user" {
                    let raw_u = if Some(idx) == last_user_idx {
                        last_user_override.unwrap_or(&m.content)
                    } else {
                        m.content.as_str()
                    };
                    let clean_u = if raw_u.contains("/no_think") {
                        raw_u.replace("/no_think", "")
                    } else {
                        raw_u.to_string()
                    };
                    let mut u_blk = String::from("<|message_user|><|content_text|>");
                    u_blk.push_str(clean_u.trim());
                    u_blk.push_str("<|end_message|>");
                    ids.extend(self.tok.encode(&u_blk));
                } else if m.role == "assistant" {
                    if !m.raw_token_ids.is_empty() {
                        if m.raw_token_ids[0] != TOK_MSG_MODEL {
                            ids.push(TOK_MSG_MODEL);
                            ids.push(TOK_CONTENT_TEXT);
                        }
                        ids.extend_from_slice(&m.raw_token_ids);
                        if m.raw_token_ids.last().copied() != Some(TOK_END_MESSAGE) {
                            ids.push(TOK_END_MESSAGE);
                        }
                    } else {
                        let trimmed = m.content.trim();
                        let mut a_blk = String::new();
                        if let (Some(ts), Some(te)) = (trimmed.find("<think>"), trimmed.find("</think>")) {
                            let th_b = trimmed[ts + 7..te].trim();
                            let tx_b = trimmed[te + 8..].trim();
                            if !th_b.is_empty() {
                                a_blk.push_str("<|message_model|><|content_thinking|>");
                                a_blk.push_str(th_b);
                                a_blk.push_str("<|end_message|>");
                            }
                            a_blk.push_str("<|message_model|><|content_text|>");
                            a_blk.push_str(tx_b);
                            a_blk.push_str("<|end_message|>");
                        } else {
                            a_blk.push_str("<|message_model|><|content_text|>");
                            a_blk.push_str(trimmed);
                            a_blk.push_str("<|end_message|>");
                        }
                        ids.extend(self.tok.encode(&a_blk));
                    }
                }
            }
            if self.inkling_effort == "0" || user_no_think {
                ids.push(TOK_MSG_MODEL);
                ids.push(TOK_CONTENT_TEXT);
            }
            return ids;
        }

        let mut ids = Vec::with_capacity(1024);
        if let Some(bos) = self.cfg.bos_id {
            ids.push(bos);
        }
        if let Some(sys_s) = eff_sys {
            ids.extend(self.tok.role_tokens("system"));
            ids.extend(self.tok.encode(sys_s));
            ids.extend(self.tok.end_turn_tokens(self.cfg.im_end_id));
        }
        for (idx, m) in self.messages.iter().enumerate() {
            if m.role == "user" || m.role == "assistant" {
                ids.extend(self.tok.role_tokens(&m.role));
                if Some(idx) == last_user_idx {
                    if let Some(ovr) = last_user_override {
                        ids.extend(self.tok.encode(ovr));
                    } else {
                        ids.extend(self.tok.encode(&m.content));
                    }
                } else {
                    ids.extend(self.tok.encode(&m.content));
                }
                ids.extend(self.tok.end_turn_tokens(self.cfg.im_end_id));
            }
        }
        ids.extend(self.tok.role_tokens("assistant"));
        let lower_name = self.cfg.model_name.to_ascii_lowercase();
        if self.cfg.thinking_template {
            if user_no_think {
                ids.extend(self.tok.encode("<think>\n\n</think>\n\n"));
            } else {
                ids.extend(self.tok.encode("<think>\n"));
            }
        } else if lower_name.contains("qwen3") && !lower_name.contains("instruct") {
            ids.extend(self.tok.encode("<think>\n\n</think>\n\n"));
        }
        ids
    }

    fn start_generation(&mut self, user_text: String, mut force_reset_cache: bool) {
        if self.streaming_rx.is_some() {
            return;
        }
        self.stop_active_tts();
        self.messages.push(ChatMsg {
            role: "user".into(),
            content: user_text.clone(),
            raw_token_ids: Vec::new(),
            stats: None,
        });

        let mut rag_override: Option<String> = None;
        let mut topic_sys_override: Option<String> = None;
        if self.rag_badge.is_some() {
            match DaemonRagClient::rag_search(&user_text, self.rag_top_k) {
                Ok(st) if !st.hits.is_empty() => {
                    let cite_list: Vec<String> = st
                        .hits
                        .iter()
                        .enumerate()
                        .map(|(i, h)| format!("`[{}]` **{}** (`{:.3}`)", i + 1, h.title, h.score))
                        .collect();
                    self.push_info(format!(
                        "📚 **RTX 3080 Wiki RAG** (`{}` • `{:.1} ms` embed • `{:.0} µs` GPU search): {}",
                        st.strategy, st.embed_ms, st.gpu_search_us, cite_list.join(" • ")
                    ));
                    rag_override = Some(format_rag_augmented_prompt(&user_text, &st.hits));
                    force_reset_cache = true;
                }
                Ok(_) => {}
                Err(e) => self.set_toast(format!("RAG search error: {}", e)),
            }
        } else if self.topic_badge.is_some() {
            match DaemonRagClient::topic_route(&user_text, self.rag_top_k, -1.0, true) {
                Ok(rt) => {
                    force_reset_cache = true;
                    if rt.matched {
                        self.topic_lora_active = true;
                        if !rt.system_prompt.is_empty() {
                            topic_sys_override = Some(rt.system_prompt.clone());
                        }
                        self.push_info(format!(
                            "⚡ **Topic LoRA**: **{}** (`cos={:.3}` • `{:.1} ms` embed • `{:.0} µs` search • `{:.2} ms` swap • `0` context tokens)",
                            rt.topic, rt.score, rt.embed_ms, rt.gpu_search_us, rt.swap_ms
                        ));
                        self.set_toast(format!("⚡ Topic LoRA: {} (cos={:.3}, swap={:.2}ms)", rt.topic, rt.score, rt.swap_ms));
                    } else {
                        self.topic_lora_active = false;
                        self.set_toast(format!("⚡ No topic LoRA matched (top cos={:.3})", rt.score));
                    }
                }
                Err(e) => {
                    self.topic_lora_active = false;
                    self.set_toast(format!("Topic route error: {}", e));
                }
            }
        }

        let prompt_ids = self.encode_conversation(rag_override.as_deref(), topic_sys_override.as_deref());
        self.total_ctx_used = prompt_ids.len();

        // Push placeholder assistant message to stream into
        self.messages.push(ChatMsg {
            role: "assistant".into(),
            content: String::new(),
            raw_token_ids: Vec::new(),
            stats: None,
        });
        self.scroll_from_bottom = 0;
        self.stream_live_tps = 0.0;
        self.stream_live_toks = 0;

        let (rx, cancel) = spawn_generation_stream(
            prompt_ids,
            self.max_tokens,
            self.temperature,
            self.spec_enabled,
            force_reset_cache,
        );
        self.streaming_rx = Some(rx);
        self.cancel_flag = Some(cancel);
    }

    fn cancel_active_stream(&mut self) {
        let mut stopped_any = false;
        if let Some(mut rec) = self.active_recorder.take() {
            rec.abort();
            stopped_any = true;
        }
        if self.asr_rx.take().is_some() {
            stopped_any = true;
        }
        if self.hands_free_talk {
            self.hands_free_talk = false;
            self.talk_rearm_at = None;
            stopped_any = true;
        }
        if let Some(c) = &self.cancel_flag {
            c.store(true, Ordering::Relaxed);
            stopped_any = true;
        }
        if self.tts_speaking.load(Ordering::Relaxed) {
            self.stop_active_tts();
            stopped_any = true;
        }
        if self.demo.state_flags().2 {
            let _ = self.demo.configure("mod", "off", 120, &self.viz_pcm);
            stopped_any = true;
        }
        if stopped_any {
            self.set_toast("Stopped active voice / synth / GPU stream cleanly");
        }
    }

    fn warm_voice_stack_bg(&self, prefer_tts: bool) {
        let asr_ref = self.asr_model_ref.clone();
        let tts_ref = self.tts_model_ref.clone();
        let is_ink = self.is_inkling;
        thread::spawn(move || {
            if is_ink {
                if prefer_tts {
                    let _ = DaemonControl::query(DaemonQuery::TtsWarm(tts_ref));
                } else {
                    let _ = DaemonControl::query(DaemonQuery::AsrWarm(asr_ref));
                }
            } else {
                let _ = DaemonControl::query(DaemonQuery::AsrWarm(asr_ref));
                let _ = DaemonControl::query(DaemonQuery::TtsWarm(tts_ref));
            }
        });
    }

    fn start_mic_recording(&mut self) {
        if self.active_recorder.is_some() || self.asr_rx.is_some() || self.streaming_rx.is_some() {
            return;
        }
        let _ = self.demo.configure("mod", "off", 120, &self.viz_pcm);
        self.stop_active_tts();
        self.talk_rearm_at = None;
        if !self.voice_enabled {
            self.voice_enabled = true;
            self.warm_voice_stack_bg(false);
        }
        match CoreAudioRecorder::start(ASR_SAMPLE_RATE) {
            Ok(rec) => {
                self.active_recorder = Some(rec);
                self.set_toast(format!(
                    "🎙 Listening on Mac mic -> 100% RTX 3080 Qwen3-ASR-{} (pause auto-sends • Enter/^R sends now • Esc cancels)",
                    self.asr_model_ref.to_uppercase()
                ));
            }
            Err(e) => {
                self.hands_free_talk = false;
                self.set_toast(format!("Mic error: {}", e));
            }
        }
    }

    fn submit_pcm_for_transcription(&mut self, pcm_16k: Vec<f32>) {
        if pcm_16k.len() < 3200 {
            self.set_toast("Voice clip too short (< 0.2s) — press Ctrl+R and speak a question");
            if self.hands_free_talk {
                self.talk_rearm_at = Some(Instant::now() + Duration::from_millis(350));
            }
            return;
        }
        let asr_ref = self.asr_model_ref.clone();
        let lang_hint = self.asr_lang_hint.clone();
        let (tx, rx) = mpsc::channel::<Result<(String, String, f64), String>>();
        thread::spawn(move || {
            let t0 = Instant::now();
            let res = DaemonStream::transcribe(&asr_ref, &lang_hint, &pcm_16k)
                .map(|(text, sj)| (text, sj, t0.elapsed().as_secs_f64() * 1000.0));
            let _ = tx.send(res);
        });
        self.asr_rx = Some(rx);
    }

    fn stop_and_submit_mic(&mut self) {
        let Some(mut rec) = self.active_recorder.take() else {
            return;
        };
        let pcm = rec.finish_and_take_samples();
        self.submit_pcm_for_transcription(pcm);
    }

    fn pump_voice_input(&mut self) -> bool {
        let mut changed = false;

        // 1. Check live microphone auto-VAD endpointing (`0.95s` trailing silence after speech)
        let should_auto_stop = self
            .active_recorder
            .as_ref()
            .map(|r| r.live_stats().auto_stop_ready)
            .unwrap_or(false);
        if should_auto_stop {
            self.stop_and_submit_mic();
            changed = true;
        }

        // 2. Check completed RTX 3080 `Qwen3-ASR-1.7B` transcription
        let mut asr_done: Option<Result<(String, String, f64), String>> = None;
        if let Some(rx) = &self.asr_rx {
            if let Ok(res) = rx.try_recv() {
                asr_done = Some(res);
            }
        }
        if let Some(res) = asr_done {
            self.asr_rx = None;
            self.hw_telemetry = query_hw_telemetry_from_daemon();
            changed = true;
            match res {
                Ok((transcript, _stats_json, stt_ms)) => {
                    let clean = transcript.trim().to_string();
                    self.last_asr_ms = stt_ms;
                    if clean.is_empty() {
                        self.set_toast(format!(
                            "🎙 No speech detected ({:.0}ms RTX 3080 STT) — press Ctrl+R to speak",
                            stt_ms
                        ));
                        if self.hands_free_talk {
                            self.talk_rearm_at = Some(Instant::now() + Duration::from_millis(350));
                        }
                    } else {
                        self.set_toast(format!("🎙 RTX 3080 STT ({:.0} ms): \"{}\"", stt_ms, clean));
                        self.voice_enabled = true;
                        self.start_generation(clean, false);
                    }
                }
                Err(e) => {
                    self.hands_free_talk = false;
                    self.set_toast(format!("RTX 3080 ASR error: {}", e));
                }
            }
        }

        // 3. Hands-free `/talk` turn-taking: once TTS finishes speaking, re-open mic after brief echo guard
        let now_speaking = self.tts_speaking.load(Ordering::Relaxed);
        if self.prev_tts_speaking && !now_speaking {
            self.hw_telemetry = query_hw_telemetry_from_daemon();
            changed = true;
        }
        if self.hands_free_talk {
            if self.prev_tts_speaking && !now_speaking && self.streaming_rx.is_none() && self.active_recorder.is_none() && self.asr_rx.is_none() {
                self.talk_rearm_at = Some(Instant::now() + Duration::from_millis(400));
                changed = true;
            }
            if let Some(when) = self.talk_rearm_at {
                if Instant::now() >= when
                    && !now_speaking
                    && self.streaming_rx.is_none()
                    && self.active_recorder.is_none()
                    && self.asr_rx.is_none()
                {
                    self.talk_rearm_at = None;
                    self.start_mic_recording();
                    changed = true;
                }
            }
        }
        self.prev_tts_speaking = now_speaking;
        changed
    }

    fn pump_stream(&mut self) -> bool {
        let mut changed = false;
        let mut finished = false;
        let mut speak_reply_after: Option<String> = None;

        if let Some(rx) = &self.streaming_rx {
            while let Ok(ev) = rx.try_recv() {
                changed = true;
                match ev {
                    StreamEvent::Token {
                        piece,
                        tok_count,
                        elapsed_s,
                        ttft_ms,
                    } => {
                        if let Some(last) = self.messages.last_mut() {
                            last.content.push_str(&piece);
                        }
                        self.stream_live_toks = tok_count;
                        self.last_ttft_ms = ttft_ms;
                        if tok_count > 1 && elapsed_s > 1e-4 {
                            self.stream_live_tps = ((tok_count - 1) as f64) / elapsed_s;
                        }
                        self.total_ctx_used += 1;
                    }
                    StreamEvent::Finished(mut st) => {
                        if st.cancelled && self.stream_live_tps > 0.0 {
                            st.decode_tok_s = self.stream_live_tps;
                        }
                        if st.decode_tok_s > 0.0 {
                            self.last_tps = st.decode_tok_s;
                            self.demo.record_turn_tps(self.last_tps);
                        }
                        if st.prefill_tok_s > 0.0 {
                            self.last_prefill_tps = st.prefill_tok_s;
                        }
                        if st.ttft_ms > 0.0 {
                            self.last_ttft_ms = st.ttft_ms;
                        }
                        if st.prompt_tokens > 0 {
                            self.total_ctx_used = st.prompt_tokens + st.completion_tokens;
                            self.last_cached_prefix = st.cached_prefix_tokens;
                        }
                        self.last_launches = st.kernel_launches;
                        self.last_spec_extra = st.spec_accepted_extra;
                        let was_cancelled = st.cancelled;
                        if let Some(last) = self.messages.last_mut() {
                            if !st.token_ids.is_empty() {
                                last.raw_token_ids = st.token_ids.clone();
                            }
                            if self.voice_enabled && !was_cancelled && !last.content.is_empty() {
                                speak_reply_after = Some(last.content.clone());
                            }
                            last.stats = Some(st);
                        }
                        finished = true;
                    }
                    StreamEvent::Error(err) => {
                        if let Some(last) = self.messages.last_mut() {
                            if last.content.is_empty() {
                                last.role = "info".into();
                                last.content = format!("Error: {}", err);
                            }
                        }
                        self.toast = Some((format!("Daemon error: {}", err), Instant::now()));
                        finished = true;
                    }
                }
            }
        }

        if finished {
            self.streaming_rx = None;
            self.cancel_flag = None;
            if self.topic_lora_active {
                let _ = DaemonControl::query(DaemonQuery::LoraClear);
                self.topic_lora_active = false;
                self.lora_badge = query_lora_badge_from_daemon();
            }
            self.hw_telemetry = query_hw_telemetry_from_daemon();
            if let Some(reply_md) = speak_reply_after {
                self.spawn_tts_for_text(&reply_md);
            }
        }
        changed
    }

    fn handle_slash_command(&mut self, cmd_line: &str, guard: Option<&mut RawTermGuard>) -> Option<bool> {
        // Returns `Some(true)` to quit TUI, `Some(false)` if handled, `None` if not a slash command
        let trimmed = cmd_line.trim();
        if !trimmed.starts_with('/') {
            return None;
        }
        let (cmd, arg) = match trimmed.split_once(char::is_whitespace) {
            Some((c, a)) => (c, a.trim()),
            None => (trimmed, ""),
        };

        match cmd {
            "/q" | "/quit" | "/exit" => return Some(true),
            "/stop" => {
                let _ = DaemonControl::stop();
                return Some(true);
            }
            "/clear" | "/reset" => {
                self.messages.clear();
                self.total_ctx_used = 0;
                self.last_cached_prefix = 0;
                self.scroll_from_bottom = 0;
                let _ = DaemonControl::query(DaemonQuery::ResetCache);
                self.set_toast("✓ Cleared chat & reset VRAM KV + ShortConv cache");
            }
            "/spec" => {
                self.spec_enabled = match arg {
                    "on" | "1" | "true" => true,
                    "off" | "0" | "false" => false,
                    _ => !self.spec_enabled,
                };
                if self.is_inkling {
                    let msg = if self.spec_enabled {
                        format!(
                            "✓ **DSpark Block-8 Speculative Decoding: ON** (`γ={}` • up to `138 tok/s` on structured JSON/code)",
                            self.inkling_gamma
                        )
                    } else {
                        "✓ **48-Layer Persistent B=1 Megakernel: ON** (`~120 tok/s` across all tasks, `2` GPU launches)".to_string()
                    };
                    self.push_info(msg.clone());
                    self.set_toast(msg);
                } else {
                    self.set_toast(if self.spec_enabled {
                        "✓ BM=4 Lossless Speculative Verify: ON (up to 260+ tok/s on code/JSON)"
                    } else {
                        "✓ Pure T=1 68-SM Persistent Megakernel: ON (~173 tok/s, 2 GPU launches)"
                    });
                }
            }
            "/effort" | "/think" => {
                if self.is_inkling {
                    let next_eff = if cmd == "/think" {
                        match arg {
                            "off" | "0" | "false" => "0",
                            "on" | "1" | "true" => "0.2",
                            "" => if self.inkling_effort == "0" { "0.2" } else { "0" },
                            other => InklingEngine::normalize_effort(other),
                        }
                    } else if arg.is_empty() {
                        if self.inkling_effort == "0" { "0.2" } else { "0" }
                    } else {
                        InklingEngine::normalize_effort(arg)
                    };
                    if next_eff != self.inkling_effort {
                        self.inkling_effort = next_eff.to_string();
                        let _ = DaemonControl::query(DaemonQuery::ResetCache);
                    }
                    let mode_desc = if self.inkling_effort == "0" {
                        "Thinking **OFF** (`effort = 0` — instant non-thinking replies at `~120 tok/s`)"
                    } else {
                        "Thinking **ON** (`<think>...</think>` 3-pass reasoning active)"
                    };
                    self.push_info(format!(
                        "✓ **Inkling Thinking Effort: `{}`** — {}. Type a prompt below to test!",
                        self.inkling_effort, mode_desc
                    ));
                    self.set_toast(format!(
                        "✓ Inkling thinking effort set to '{}' (0 = instant ~120 tok/s • 0.2 = reasoning • 0.9 = deep)",
                        self.inkling_effort
                    ));
                } else {
                    self.set_toast("Tip: append `/no_think` to any prompt to skip thinking on Qwen3");
                }
            }
            "/gamma" => {
                if let Ok(g) = arg.parse::<usize>() {
                    let gc = g.clamp(1, 7);
                    match DaemonControl::query(DaemonQuery::InklingSetGamma(gc)) {
                        Ok(_) => {
                            self.inkling_gamma = gc;
                            self.hw_telemetry = query_hw_telemetry_from_daemon();
                            self.push_info(format!(
                                "✓ **DSpark Draft Depth: `γ={}`** (speculative verify currently **{}** — use `/spec on` to enable)",
                                gc, if self.spec_enabled { "ON" } else { "OFF" }
                            ));
                            self.set_toast(format!("✓ DSpark draft depth set to γ={} (use `/spec on` to enable)", gc));
                        }
                        Err(e) => self.set_toast(format!("Gamma update error: {}", e)),
                    }
                } else {
                    self.set_toast(format!("Current DSpark γ={} (usage: /gamma 1..7)", self.inkling_gamma));
                }
            }
            "/temp" => {
                if let Ok(v) = arg.parse::<f32>() {
                    self.temperature = v.clamp(0.0, 2.0);
                    self.set_toast(format!("✓ Sampling temperature set to {:.2}", self.temperature));
                } else {
                    self.set_toast(format!("Current temperature: {:.2} (usage: /temp 0.0 .. 2.0)", self.temperature));
                }
            }
            "/max" => {
                if let Ok(n) = arg.parse::<usize>() {
                    self.max_tokens = n.clamp(16, 4096);
                    self.set_toast(format!("✓ Max completion tokens set to {}", self.max_tokens));
                } else {
                    self.set_toast(format!("Current max_tokens: {} (usage: /max 1024)", self.max_tokens));
                }
            }
            "/sys" | "/system" => {
                if arg == "clear" {
                    self.system_prompt.clear();
                    self.set_toast("✓ Cleared system prompt");
                } else if !arg.is_empty() {
                    self.system_prompt = arg.to_string();
                    self.set_toast("✓ Updated system prompt");
                } else {
                    let cur = if self.system_prompt.is_empty() { "(none)" } else { &self.system_prompt };
                    self.push_info(format!("**System Prompt**: `{}`\nUse `/sys <prompt>` to set or `/sys clear` to remove.", cur));
                }
            }
            "/mouse" => {
                self.mouse_capture = !self.mouse_capture;
                if let Some(g) = guard {
                    g.set_mouse(self.mouse_capture);
                }
                self.set_toast(if self.mouse_capture {
                    "✓ Mouse wheel scrolling ON (hold Option/Alt to drag-select text)"
                } else {
                    "✓ Mouse wheel capture OFF (native terminal click-select active)"
                });
            }
            "/copy" => {
                if let Some(m) = self.messages.iter().rev().find(|m| m.role == "assistant") {
                    let target = if arg == "code" {
                        extract_last_fenced_code(&m.content).unwrap_or_else(|| m.content.clone())
                    } else {
                        m.content.clone()
                    };
                    match copy_to_macos_clipboard(&target) {
                        Ok(()) => self.set_toast(format!("✓ Copied {} chars to macOS clipboard", target.len())),
                        Err(e) => self.set_toast(e),
                    }
                } else {
                    self.set_toast("No assistant reply to copy yet");
                }
            }
            "/verify" => {
                let body = format_verify_markdown(
                    self.is_inkling, &self.cfg, &self.gguf_path, self.gguf_size_gb, self.active_gb_per_tok,
                );
                self.push_info(body);
            }
            "/bench" => {
                let (en_spec, prompt) = select_bench_prompt(self.is_inkling, arg);
                self.spec_enabled = en_spec;
                self.start_generation(prompt, true);
            }
            "/speak" => {
                self.voice_enabled = match arg {
                    "on" | "1" | "true" => true,
                    "off" | "0" | "false" => false,
                    _ => !self.voice_enabled,
                };
                if self.voice_enabled {
                    self.warm_voice_stack_bg(true);
                    self.set_toast(format!(
                        "🔊 Live RTX 3080 Voice ON (Qwen3-TTS {} Q8_0 • speaker='{}' • Ctrl+R to talk)",
                        self.tts_model_ref, self.voice_speaker
                    ));
                } else {
                    self.hands_free_talk = false;
                    self.talk_rearm_at = None;
                    self.stop_active_tts();
                    self.set_toast("🔇 Live RTX 3080 Voice OFF");
                }
            }
            "/listen" | "/mic" => {
                if !arg.is_empty() {
                    // Support `/listen <path.wav>` to transcribe an audio file straight into the live TUI chat + voice loop!
                    match read_wav_mono_16k(std::path::Path::new(arg)) {
                        Ok(pcm) => {
                            self.voice_enabled = true;
                            self.submit_pcm_for_transcription(pcm);
                        }
                        Err(e) => self.set_toast(format!("WAV load error: {}", e)),
                    }
                } else if self.active_recorder.is_some() {
                    self.stop_and_submit_mic();
                } else {
                    self.start_mic_recording();
                }
            }
            "/talk" => {
                self.hands_free_talk = match arg {
                    "on" | "1" | "true" => true,
                    "off" | "0" | "false" => false,
                    _ => !self.hands_free_talk,
                };
                if self.hands_free_talk {
                    self.voice_enabled = true;
                    self.warm_voice_stack_bg(false);
                    self.start_mic_recording();
                    self.set_toast(
                        "🎙🔊 Hands-Free RTX 3080 Voice Loop ON (speak naturally • auto-listens after each reply • Esc stops)",
                    );
                } else {
                    self.talk_rearm_at = None;
                    if let Some(mut rec) = self.active_recorder.take() {
                        rec.abort();
                    }
                    self.set_toast("Hands-Free Voice Loop OFF (press Ctrl+R anytime for single-turn voice)");
                }
            }
            "/say" => {
                if arg.is_empty() {
                    if let Some(m) = self.messages.iter().rev().find(|m| m.role == "assistant") {
                        let text = m.content.clone();
                        self.spawn_tts_for_text(&text);
                        self.set_toast(format!("🔊 Speaking last reply via RTX 3080 ('{}')...", self.voice_speaker));
                    } else {
                        self.set_toast("Usage: /say <text> (or /say after an assistant reply)");
                    }
                } else {
                    self.spawn_tts_for_text(arg);
                    self.set_toast(format!("🔊 Speaking on RTX 3080 ('{}')...", self.voice_speaker));
                }
            }
            "/voice" | "/speaker" => {
                if arg.is_empty() {
                    self.set_toast(format!(
                        "Current voice: '{}' (presets: ryan, serena, vivian, aiden, uncle_fu, ono_anna, sohee, eric, dylan)",
                        self.voice_speaker
                    ));
                } else {
                    self.voice_speaker = arg.to_ascii_lowercase();
                    self.set_toast(format!("✓ TTS voice preset set to '{}'", self.voice_speaker));
                }
            }
            "/lora" => {
                if arg.is_empty() || arg == "status" {
                    let st = DaemonControl::query(DaemonQuery::LoraStatus).unwrap_or_else(|e| format!("Error: {}", e));
                    self.lora_badge = query_lora_badge_from_daemon();
                    self.push_info(format!("**Fused Rank-r LoRA Status**: `{}`\nUse `/lora <adapter_path>` to hot-swap (`< 1 ms`) or `/lora off` to clear.", st));
                } else if matches!(arg, "off" | "clear" | "none") {
                    match DaemonControl::query(DaemonQuery::LoraClear) {
                        Ok(_) => {
                            self.lora_badge = None;
                            self.set_toast("✓ Cleared active LoRA adapter (0.0% base overhead)");
                        }
                        Err(e) => self.set_toast(format!("LoRA clear error: {}", e)),
                    }
                } else {
                    let path_arg = arg.strip_prefix("load ").map(|s| s.trim()).unwrap_or(arg);
                    match DaemonControl::query(DaemonQuery::LoraLoad { path: path_arg.to_string(), scale: 1.0 }) {
                        Ok(res) => {
                            self.lora_badge = query_lora_badge_from_daemon();
                            self.set_toast(format!("✓ Hot-swapped LoRA: {}", res));
                        }
                        Err(e) => self.set_toast(format!("LoRA load failed: {}", e)),
                    }
                }
                self.hw_telemetry = query_hw_telemetry_from_daemon();
            }
            "/rag" => {
                if arg.is_empty() || arg == "status" {
                    let st = DaemonControl::query(DaemonQuery::RagStatus).unwrap_or_else(|e| format!("Error: {}", e));
                    self.rag_badge = query_rag_badge_from_daemon();
                    self.push_info(format!(
                        "**RTX 3080 Same-Model VectorDB Status**: `{}`\nUse `/rag wiki` (or `/rag <file.qvec>`) to enable live Wikipedia RAG, or `/rag off` to disable.",
                        st
                    ));
                } else if matches!(arg, "off" | "clear" | "none") {
                    match DaemonControl::query(DaemonQuery::RagClear) {
                        Ok(_) => {
                            self.rag_badge = None;
                            self.set_toast("✓ Disabled RTX 3080 RAG VectorDB");
                        }
                        Err(e) => self.set_toast(format!("RAG clear error: {}", e)),
                    }
                } else {
                    let resolved = if matches!(arg, "on" | "wiki" | "wikipedia") {
                        "data/wikipedia_1k_best.qvec"
                    } else {
                        arg.strip_prefix("load ").map(|s| s.trim()).unwrap_or(arg)
                    };
                    match DaemonControl::query(DaemonQuery::RagLoad(resolved.to_string())) {
                        Ok(res) => {
                            self.rag_badge = query_rag_badge_from_daemon();
                            self.topic_badge = None;
                            self.set_toast(format!("📚 Loaded RTX 3080 VectorDB: {}", res));
                        }
                        Err(e) => self.set_toast(format!("RAG load failed: {}", e)),
                    }
                }
                self.hw_telemetry = query_hw_telemetry_from_daemon();
            }
            "/topic" => {
                if arg.is_empty() || arg == "status" {
                    let st = DaemonControl::query(DaemonQuery::TopicStatus).unwrap_or_else(|e| format!("Error: {}", e));
                    self.topic_badge = query_topic_badge_from_daemon();
                    self.push_info(format!(
                        "**RTX 3080 Sharded Topic LoRA Bank Status**: `{}`\nUse `/topic wiki` (or `/topic <bank_dir>`) to enable bare-prompt Topic LoRA routing, `/topic list` to list topics, or `/topic off` to disable.",
                        st
                    ));
                } else if arg == "list" {
                    let lst = DaemonControl::query(DaemonQuery::TopicList).unwrap_or_else(|e| format!("Error: {}", e));
                    self.push_info(format!("**Loaded Topic LoRA Bank**: `{}`", lst));
                } else if matches!(arg, "off" | "clear" | "none") {
                    match DaemonControl::query(DaemonQuery::TopicClear) {
                        Ok(_) => {
                            self.topic_badge = None;
                            self.topic_lora_active = false;
                            self.lora_badge = query_lora_badge_from_daemon();
                            self.set_toast("✓ Disabled RTX 3080 Topic LoRA Bank");
                        }
                        Err(e) => self.set_toast(format!("Topic clear error: {}", e)),
                    }
                } else {
                    let resolved = if matches!(arg, "on" | "wiki" | "wikipedia") {
                        "checkpoints/wiki-topic-loras"
                    } else {
                        arg.strip_prefix("load ").map(|s| s.trim()).unwrap_or(arg)
                    };
                    match DaemonControl::query(DaemonQuery::TopicLoad(resolved.to_string())) {
                        Ok(res) => {
                            self.topic_badge = query_topic_badge_from_daemon();
                            self.rag_badge = None;
                            self.set_toast(format!("⚡ Loaded Topic LoRA Bank: {}", res));
                        }
                        Err(e) => self.set_toast(format!("Topic load failed: {}", e)),
                    }
                }
                self.hw_telemetry = query_hw_telemetry_from_daemon();
            }
            "/fold" => {
                self.expand_thinking = match arg {
                    "on" | "1" | "expand" | "open" => true,
                    "off" | "0" | "collapse" | "close" => false,
                    _ => !self.expand_thinking,
                };
                self.set_toast(if self.expand_thinking {
                    "Thinking traces: expanded (Ctrl+O to collapse)"
                } else {
                    "Thinking traces: collapsed to summary (Ctrl+O to expand)"
                });
            }
            "/inspector" | "/sidebar" | "/fx" => {
                let msg = self.demo.configure("sidebar", arg, term_size().0, &self.viz_pcm);
                self.set_toast(msg);
            }
            "/demo" => {
                let msg = self.demo.configure("demo_open", arg, term_size().0, &self.viz_pcm);
                if !msg.is_empty() {
                    self.set_toast(msg);
                }
            }
            "/theme" | "/palette" => {
                let msg = self.demo.configure("theme", arg, term_size().0, &self.viz_pcm);
                self.set_toast(msg);
            }
            "/mod" | "/music" | "/chiptune" => {
                self.stop_active_tts();
                let msg = self.demo.configure("mod", arg, term_size().0, &self.viz_pcm);
                self.set_toast(msg);
            }
            "/help" | _ => {
                self.push_info(format_help_markdown(self.is_inkling));
            }
        }
        Some(false)
    }

    fn draw(&mut self, scr: &mut ScreenBuf, w: usize, h: usize) -> Option<(usize, usize)> {
        let pal = self.demo.palette();
        let (theme_lbl, showcase_open, mod_on) = self.demo.state_flags();

        if showcase_open {
            self.demo.draw_megademo(scr, w, h, &self.cfg.model_name, self.active_sms);
            return None;
        }

        scr.resize_and_clear(w, h, pal.bg_void);
        let sidebar_on = self.demo.sidebar_visible(w);
        let sidebar_w = if sidebar_on { 34 } else { 0 };
        let main_w = w.saturating_sub(sidebar_w).max(24);

        let inner_comp_w = main_w.saturating_sub(6).max(10);
        let (comp_lines, cur_comp_row, cur_comp_col) = self.composer.wrapped_lines(inner_comp_w);
        let vis_comp_rows = comp_lines.len().clamp(1, 6);
        let comp_scroll = cur_comp_row.saturating_add(1).saturating_sub(vis_comp_rows);

        let footer_y = h.saturating_sub(1);
        let comp_bot_y = footer_y.saturating_sub(1);
        let comp_top_y = comp_bot_y.saturating_sub(vis_comp_rows + 1);
        let chat_top_y = 3usize;
        let chat_h = comp_top_y.saturating_sub(chat_top_y);

        // 1. Main Chat Viewport (or Calm Session Brief when empty)
        if !self.messages.is_empty() {
            let mut all_lines: Vec<RenderedLine> = Vec::with_capacity(256);
            let last_asst_idx = self.messages.iter().rposition(|m| m.role == "assistant");
            if last_asst_idx.is_none() {
                self.last_think_words = 0;
                self.last_answer_words = 0;
            }

            for (idx, msg) in self.messages.iter().enumerate() {
                if idx > 0 {
                    all_lines.push(RenderedLine { spans: Vec::new(), row_bg: pal.bg_void });
                }
                let is_last_streaming = self.streaming_rx.is_some() && idx + 1 == self.messages.len();
                let mut hdr_spans = vec![Span { text: "  ".into(), style: Style::new(pal.text_hi, pal.bg_void) }];

                if msg.role == "user" {
                    hdr_spans.push(Span {
                        text: "▍ You".into(),
                        style: Style::new(pal.accent_user, pal.bg_void).bold(),
                    });
                } else if msg.role == "assistant" {
                    hdr_spans.push(Span {
                        text: "▍ ".into(),
                        style: Style::new(pal.accent_bot, pal.bg_void).bold(),
                    });
                    hdr_spans.push(Span {
                        text: self.cfg.model_name.clone(),
                        style: Style::new(pal.text_hi, pal.bg_void).bold(),
                    });
                    if is_last_streaming {
                        const FRAMES: [char; 4] = ['·', '✢', '✳', '◈'];
                        let sp = FRAMES[(self.spinner_tick / 2) % FRAMES.len()];
                        hdr_spans.push(Span {
                            text: format!("  {} {} toks · {:.1} tok/s", sp, self.stream_live_toks, self.stream_live_tps),
                            style: Style::new(pal.grad_c, pal.bg_void),
                        });
                    } else if let Some(st) = &msg.stats {
                        let eff_bw = st.decode_tok_s * self.active_gb_per_tok;
                        let cache_note = if st.cached_prefix_tokens > 0 {
                            format!(" · +{} cached", st.cached_prefix_tokens)
                        } else {
                            String::new()
                        };
                        let spec_note = if st.spec_accepted_extra > 0 {
                            format!(" · +{} spec", st.spec_accepted_extra)
                        } else {
                            String::new()
                        };
                        let status_note = if st.cancelled { " · stopped" } else { "" };
                        hdr_spans.push(Span {
                            text: format!(
                                "  · {:.1} tok/s · {:.0} GB/s · TTFT {:.0}ms{}{}{}",
                                st.decode_tok_s, eff_bw, st.ttft_ms, cache_note, spec_note, status_note
                            ),
                            style: Style::new(pal.text_dim, pal.bg_void),
                        });
                    }
                } else {
                    let is_err = msg.content.starts_with("Error:");
                    hdr_spans.push(Span {
                        text: if is_err { "▍ Error".into() } else { "▍ System".into() },
                        style: Style::new(if is_err { FG_CORAL } else { pal.grad_b }, pal.bg_void).bold(),
                    });
                }
                all_lines.push(RenderedLine { spans: hdr_spans, row_bg: pal.bg_void });

                let rendered = if msg.role == "assistant" {
                    let (rls, th_w, ans_w) = format_assistant_markdown(
                        &msg.content,
                        main_w.saturating_sub(2),
                        is_last_streaming,
                        self.expand_thinking,
                        pal.bg_void,
                        pal.text_muted,
                        pal.text_dim,
                        pal.grad_a,
                    );
                    if Some(idx) == last_asst_idx {
                        self.last_think_words = th_w;
                        self.last_answer_words = ans_w;
                    }
                    rls
                } else {
                    format_markdown_block(&msg.content, main_w.saturating_sub(2), msg.role == "user")
                };

                for mut rl in rendered {
                    if rl.row_bg == terminal::BG_MAIN {
                        rl.row_bg = pal.bg_void;
                        for sp in &mut rl.spans {
                            if sp.style == Style::new(terminal::FG_TEXT, terminal::BG_MAIN) {
                                sp.style = Style::new(pal.text_hi, pal.bg_void);
                            }
                        }
                    }
                    all_lines.push(rl);
                }
            }

            let max_scroll = all_lines.len().saturating_sub(chat_h);
            if self.scroll_from_bottom > max_scroll {
                self.scroll_from_bottom = max_scroll;
            }
            let end_idx = all_lines.len().saturating_sub(self.scroll_from_bottom);
            let start_idx = end_idx.saturating_sub(chat_h);

            for (row_off, rline) in all_lines[start_idx..end_idx].iter().enumerate() {
                let y = chat_top_y + row_off;
                if rline.row_bg != pal.bg_void {
                    scr.fill_row_range_help(0, y, main_w, rline.row_bg);
                }
                let mut rx = 0usize;
                for sp in &rline.spans {
                    rx = scr.put_str(rx, y, main_w.saturating_sub(1), &sp.text, sp.style);
                }
            }

            if self.scroll_from_bottom > 0 && chat_h > 0 {
                let pill = format!(" ↑ +{} lines (End) ", self.scroll_from_bottom);
                let px = main_w.saturating_sub(pill.chars().count() + 2);
                scr.put_str(px, comp_top_y.saturating_sub(1), main_w, &pill, Style::new(pal.bg_void, pal.grad_b).bold());
            }
        }

        // 2. Rounded Prompt Composer (`╭─ ... ─╮`)
        let mic_stats = self.active_recorder.as_ref().map(|r| r.live_stats());
        let is_transcribing = self.asr_rx.is_some();
        let is_busy = self.streaming_rx.is_some() || is_transcribing;
        let border_fg = if let Some(ref ms) = mic_stats {
            if ms.speech_started { FG_GREEN } else { FG_CORAL }
        } else if is_transcribing {
            FG_CYAN
        } else if is_busy {
            pal.grad_c
        } else {
            pal.border_mid
        };
        let border_st = Style::new(border_fg, pal.bg_deck);
        let box_w = main_w.saturating_sub(2);

        let title = if mic_stats.is_some() {
            format!(
                "─ ● Recording 16 kHz -> RTX 3080 ASR-{} (Enter/^R send · Esc cancel) ",
                self.asr_model_ref.to_uppercase()
            )
        } else if is_transcribing {
            format!("─ ◈ Transcribing on RTX 3080 (Qwen3-ASR-{})... ", self.asr_model_ref.to_uppercase())
        } else if is_busy {
            "─ ◈ Generating on GA102 68 SMs (Esc stop) ".to_string()
        } else {
            format!("─ ▸ Prompt · {} ", self.cfg.model_name)
        };
        scr.fill_row_range_help(1, comp_top_y, box_w, pal.bg_deck);
        scr.put_char(1, comp_top_y, '╭', border_st);
        let mut tx = scr.put_str(2, comp_top_y, main_w.saturating_sub(2), &title, Style::new(border_fg, pal.bg_deck).bold());
        while tx < main_w.saturating_sub(2) {
            tx += scr.put_char(tx, comp_top_y, '─', Style::new(pal.border_dim, pal.bg_deck));
        }
        scr.put_char(main_w.saturating_sub(2), comp_top_y, '╮', border_st);

        for r in 0..vis_comp_rows {
            let y = comp_top_y + 1 + r;
            scr.fill_row_range_help(1, y, box_w, pal.bg_deck);
            scr.put_char(1, y, '│', Style::new(pal.border_dim, pal.bg_deck));
            scr.put_char(main_w.saturating_sub(2), y, '│', Style::new(pal.border_dim, pal.bg_deck));
            let line_idx = comp_scroll + r;
            if let Some(ref ms) = mic_stats {
                if r == 0 {
                    let level = ((ms.rms * 14.0).sqrt()).clamp(0.0, 1.0) as f64;
                    let lit = ((level * 12.0).round() as usize).min(12);
                    let bar = format!("● VU [{}{}] {:4.1}s ", "█".repeat(lit), "░".repeat(12 - lit), ms.duration_sec);
                    let st = Style::new(if ms.speech_started { FG_GREEN } else { pal.text_hi }, pal.bg_deck).bold();
                    let nx = scr.put_str(3, y, main_w.saturating_sub(3), &bar, st);
                    let sx = self.demo.draw_voice_spectrum(scr, nx + 1, y, main_w.saturating_sub(4), pal.bg_deck);
                    let hint = if ms.speech_started { "  pause ~1s auto-sends" } else { "  speak now..." };
                    scr.put_str(sx + 1, y, main_w.saturating_sub(3), hint, Style::new(pal.text_muted, pal.bg_deck));
                }
            } else if is_transcribing && r == 0 {
                let stt_msg = format!(
                    "Running GPU STFT/Mel -> AuT Encoder -> Qwen3-ASR-{} on RTX 3080...",
                    self.asr_model_ref.to_uppercase()
                );
                scr.put_str(3, y, main_w.saturating_sub(3), &stt_msg, Style::new(FG_CYAN, pal.bg_deck));
            } else if self.composer.is_empty() && r == 0 && !is_busy {
                scr.put_str(
                    3,
                    y,
                    main_w.saturating_sub(3),
                    "Ask anything, type / for commands, or press Ctrl+R to speak...",
                    Style::new(pal.text_dim, pal.bg_deck),
                );
            } else if let Some(ln) = comp_lines.get(line_idx) {
                scr.put_str(3, y, main_w.saturating_sub(3), ln, Style::new(pal.text_hi, pal.bg_deck));
            }
        }

        scr.fill_row_range_help(1, comp_bot_y, box_w, pal.bg_deck);
        scr.put_char(1, comp_bot_y, '╰', border_st);
        for bx in 2..main_w.saturating_sub(2) {
            scr.put_char(bx, comp_bot_y, '─', Style::new(pal.border_dim, pal.bg_deck));
        }
        scr.put_char(main_w.saturating_sub(2), comp_bot_y, '╯', border_st);

        // 3. Snapshot Telemetry -> Empty-State Brief, Top Instrument Bar, Right Inspector, & Slash Palette
        let snap = self.snapshot();
        if self.messages.is_empty() {
            self.demo.draw_session_brief(scr, 0, chat_top_y, main_w, chat_h, &snap);
        }

        // Row 0 & Row 1: Top Instrument Header + Row 2 Copper Hairline Rule
        scr.fill_row(0, pal.bg_header);
        scr.fill_row(1, pal.bg_header);
        let brand = " ▄▀ RTX ";
        let mut x = 1;
        for (i, ch) in brand.chars().enumerate() {
            let col = copper_gradient(&pal, (i as f32) / 8.0);
            x += scr.put_char(x, 0, ch, Style::new(col, pal.bg_header).bold());
        }
        x = scr.put_str(x + 1, 0, w - 1, &self.cfg.model_name, Style::new(pal.text_hi, pal.bg_header).bold());
        let arch_tag = if self.is_inkling {
            format!(" · 48L 128×8 MoE ({:.2}G)", self.gguf_size_gb)
        } else {
            format!(" · {:.2}G GGUF", self.gguf_size_gb)
        };
        x = scr.put_str(x, 0, w - 1, &arch_tag, Style::new(pal.text_muted, pal.bg_header));

        let dot_fg = if self.daemon_online { FG_GREEN } else { FG_CORAL };
        let gpu_tag = format!(" ● GA102 {} SMs · {:.1}ms ", self.active_sms, self.ping_ms);
        let vram_cells = 10usize;
        let rhs_need = gpu_tag.chars().count() + vram_cells + 7;
        if w > x + rhs_need + 2 {
            let vx = w - rhs_need - 1;
            let bx = scr.put_str(vx, 0, w - 1, "VRAM ", Style::new(pal.text_dim, pal.bg_header));
            let ax = self.demo.draw_vram_bar(scr, bx, 0, vram_cells, pal.bg_header, &snap);
            scr.put_str(ax + 1, 0, w - 1, &gpu_tag, Style::new(dot_fg, pal.bg_header).bold());
        }

        // Row 1: Mode, Effort, Voice/Adapter pills, Context counter, & Roofline Sparkline
        let mut x1 = 2;
        let (mode_lbl, mode_bg) = if self.is_inkling {
            if self.spec_enabled {
                (format!(" DSpark γ={} ", self.inkling_gamma), BG_BADGE_ON)
            } else {
                (" 48L B=1 ".to_string(), BG_BADGE_T1)
            }
        } else if self.spec_enabled {
            (" Spec BM=4 ".to_string(), BG_BADGE_ON)
        } else {
            (" T=1 MK ".to_string(), BG_BADGE_T1)
        };
        x1 = scr.put_str(x1, 1, w - 1, &mode_lbl, Style::new(pal.text_hi, mode_bg).bold());
        if self.is_inkling {
            let eff_lbl = if self.inkling_effort == "0" {
                " effort 0 · fast ".to_string()
            } else {
                format!(" effort {} ", self.inkling_effort)
            };
            let eff_fg = if self.inkling_effort == "0" { FG_GREEN } else { pal.grad_a };
            x1 = scr.put_str(x1 + 1, 1, w - 1, &eff_lbl, Style::new(eff_fg, pal.bg_surface).bold());
        }
        if mod_on {
            x1 = scr.put_str(x1 + 1, 1, w - 1, " ♫ MOD ", Style::new(pal.bg_void, pal.grad_b).bold());
        }
        if self.active_recorder.is_some() {
            x1 = scr.put_str(x1 + 1, 1, w - 1, " ● REC ", Style::new(pal.bg_void, FG_GREEN).bold());
        } else if self.asr_rx.is_some() {
            x1 = scr.put_str(x1 + 1, 1, w - 1, " ◈ STT ", Style::new(pal.bg_void, FG_CYAN).bold());
        } else if snap.is_speaking {
            let lbl = if snap.last_ttfa_ms > 0.0 {
                format!(" ♫ TTS {:.0}ms ", snap.last_ttfa_ms)
            } else {
                " ♫ SYNTH ".to_string()
            };
            x1 = scr.put_str(x1 + 1, 1, w - 1, &lbl, Style::new(pal.bg_void, FG_AMBER).bold());
        } else if self.hands_free_talk {
            x1 = scr.put_str(
                x1 + 1,
                1,
                w - 1,
                &format!(" ● TALK:{} ", self.voice_speaker),
                Style::new(pal.bg_void, FG_GREEN).bold(),
            );
        } else if self.voice_enabled {
            x1 = scr.put_str(
                x1 + 1,
                1,
                w - 1,
                &format!(" ♫ {} ", self.voice_speaker),
                Style::new(pal.grad_c, pal.bg_surface).bold(),
            );
        }
        if let Some(tb) = &self.topic_badge {
            x1 = scr.put_str(x1 + 1, 1, w - 1, tb, Style::new(pal.bg_void, FG_AMBER).bold());
        } else if let Some(rb) = &self.rag_badge {
            x1 = scr.put_str(x1 + 1, 1, w - 1, rb, Style::new(pal.bg_void, FG_CYAN).bold());
        }
        if let Some(lb) = &self.lora_badge {
            x1 = scr.put_str(x1 + 1, 1, w - 1, lb, Style::new(pal.bg_void, FG_PURPLE).bold());
        }

        let ctx_str = format!(
            "  ctx {}/{} (+{})",
            self.total_ctx_used, self.cfg.max_ctx, self.last_cached_prefix
        );
        x1 = scr.put_str(x1, 1, w - 1, &ctx_str, Style::new(pal.text_muted, pal.bg_header));

        if snap.active_tps > 0.0 {
            let eff_bw = snap.active_tps * self.active_gb_per_tok;
            let sp_str = format!("{:.1} tok/s · {:.0} GB/s ", snap.active_tps, eff_bw);
            let spark_w = 8usize;
            let sx = w.saturating_sub(sp_str.chars().count() + spark_w + 2);
            if sx > x1 + 2 {
                let nx = scr.put_str(sx, 1, w - 1, &sp_str, Style::new(pal.grad_c, pal.bg_header).bold());
                self.demo.draw_tps_sparkline(scr, nx, 1, spark_w, pal.bg_header, snap.active_tps);
            }
        }

        self.demo.draw_copper_rule(scr, w, 2, is_busy);

        // Right-Side Hardware & Session Telemetry Inspector
        if sidebar_on {
            self.demo.draw_inspector_sidebar(
                scr,
                main_w,
                chat_top_y,
                sidebar_w,
                footer_y.saturating_sub(chat_top_y),
                &snap,
            );
        }

        // Interactive Slash-Command Autocomplete Popup (`/` prefix)
        if !is_busy && mic_stats.is_none() {
            let comp_txt = self.composer.as_text();
            if comp_txt.starts_with('/') && !comp_txt.contains(char::is_whitespace) {
                let items = filter_slash_commands(&comp_txt, &snap, theme_lbl, sidebar_on);
                if !items.is_empty() {
                    let sel = self.slash_sel.min(items.len() - 1);
                    self.demo.draw_slash_palette(scr, 2, comp_top_y, main_w.saturating_sub(4), &items, sel);
                }
            }
        }

        // 4. Calm Static Status & Keybind Footer (`0` distracting scrolling marquee!)
        let show_toast = self
            .toast
            .as_ref()
            .filter(|(_, t)| t.elapsed() < Duration::from_secs(4))
            .map(|(m, _)| m.clone());
        if show_toast.is_none() {
            self.toast = None;
        }
        scr.fill_row(footer_y, pal.bg_footer);
        let lhs_end = if let Some(msg) = show_toast {
            scr.put_str(2, footer_y, w.saturating_sub(2), &msg, Style::new(FG_GREEN, pal.bg_footer).bold())
        } else {
            let state_txt = if is_busy {
                format!("◈ Active · {}", theme_lbl)
            } else {
                format!("● Ready · {}", theme_lbl)
            };
            scr.put_str(2, footer_y, w.saturating_sub(2), &state_txt, Style::new(pal.text_muted, pal.bg_footer))
        };
        let keys = "^R voice · ^O think · ^B inspector · ^S spec · ^T theme · / cmds · ^Q quit ";
        let kx = w.saturating_sub(keys.chars().count());
        if kx > lhs_end + 2 {
            scr.put_str(kx, footer_y, w, keys, Style::new(pal.text_dim, pal.bg_footer));
        }

        if is_busy || mic_stats.is_some() {
            None
        } else {
            let cy = comp_top_y + 1 + (cur_comp_row.saturating_sub(comp_scroll));
            let cx = (3 + cur_comp_col).min(main_w.saturating_sub(3));
            Some((cx, cy))
        }
    }

    fn run_interactive_loop(&mut self, voice_start: bool) -> Result<(), String> {
        if voice_start {
            self.voice_enabled = true;
            self.warm_voice_stack_bg(false);
            self.set_toast("🎙🔊 RTX 3080 Voice Active — press Ctrl+R to speak (or /talk for hands-free)");
        }
        let mut guard = RawTermGuard::enter()?;
        let (w, h) = term_size();
        let mut scr = ScreenBuf::new(w, h);
        let mut paste_acc: Option<String> = None;
        let mut stdout = io::stdout();
        let mut needs_redraw = true;
        let mut last_ping_check = Instant::now();

        loop {
            let (tw, th) = term_size();
            if (tw, th) != scr.dims() {
                needs_redraw = true;
            }

            if self.pump_voice_input() {
                needs_redraw = true;
            }
            if self.pump_stream() {
                needs_redraw = true;
            }

            let (_, showcase_open, mod_on) = self.demo.state_flags();
            let gpu_busy = self.streaming_rx.is_some() || self.asr_rx.is_some();
            let anim_changed = if let Some(ref rec) = self.active_recorder {
                rec.recent_samples(256, &mut self.mic_viz_scratch);
                self.demo.step(Some((&self.mic_viz_scratch, ASR_SAMPLE_RATE)), gpu_busy)
            } else if self.tts_speaking.load(Ordering::Relaxed) || mod_on {
                self.mic_viz_scratch.clear();
                if let Ok(ring) = self.viz_pcm.lock() {
                    let start = ring.len().saturating_sub(256);
                    self.mic_viz_scratch.extend_from_slice(&ring[start..]);
                }
                self.demo.step(Some((&self.mic_viz_scratch, TTS_SAMPLE_RATE)), gpu_busy)
            } else {
                self.demo.step(None, gpu_busy)
            };
            if anim_changed {
                needs_redraw = true;
            }

            if gpu_busy
                || self.tts_speaking.load(Ordering::Relaxed)
                || self.active_recorder.is_some()
                || mod_on
            {
                self.spinner_tick = self.spinner_tick.wrapping_add(1);
                needs_redraw = true;
            }

            if !gpu_busy
                && self.active_recorder.is_none()
                && last_ping_check.elapsed() > Duration::from_secs(5)
            {
                let tp = Instant::now();
                if DaemonControl::ping().is_ok() {
                    self.ping_ms = tp.elapsed().as_secs_f64() * 1000.0;
                    self.daemon_online = true;
                    self.hw_telemetry = query_hw_telemetry_from_daemon();
                } else {
                    self.daemon_online = false;
                }
                last_ping_check = Instant::now();
                needs_redraw = true;
            }

            if needs_redraw {
                let cursor = self.draw(&mut scr, tw, th);
                let frame = scr.render_ansi(cursor);
                stdout.write_all(frame).map_err(|e| e.to_string())?;
                stdout.flush().map_err(|e| e.to_string())?;
                needs_redraw = false;
            }

            let poll_ms = if showcase_open
                || mod_on
                || gpu_busy
                || self.active_recorder.is_some()
                || self.tts_speaking.load(Ordering::Relaxed)
                || self.talk_rearm_at.is_some()
            {
                16
            } else {
                50
            };
            let evs = poll_input_events(poll_ms, &mut paste_acc);
            if evs.is_empty() {
                continue;
            }
            needs_redraw = true;

            for ev in evs {
                if self.demo.state_flags().1 {
                    match ev {
                        InputEvent::CtrlQ => {
                            let _ = self.demo.configure("stop_all", "", tw, &self.viz_pcm);
                            self.cancel_active_stream();
                            return Ok(());
                        }
                        InputEvent::Esc | InputEvent::CtrlC | InputEvent::Char('q') | InputEvent::Char('Q') => {
                            let msg = self.demo.configure("demo_open", "off", tw, &self.viz_pcm);
                            self.set_toast(msg);
                        }
                        InputEvent::Char(c @ ('1'..='4')) => {
                            let _ = self.demo.configure("demo_scene", &c.to_string(), tw, &self.viz_pcm);
                        }
                        InputEvent::Char(' ') | InputEvent::Right | InputEvent::Enter => {
                            let _ = self.demo.configure("demo_scene", "next", tw, &self.viz_pcm);
                        }
                        InputEvent::Char('t') | InputEvent::Char('T') | InputEvent::CtrlT => {
                            let _ = self.demo.configure("theme_cycle", "", tw, &self.viz_pcm);
                        }
                        InputEvent::Char('m') | InputEvent::Char('M') => {
                            let _ = self.demo.configure("mod", "", tw, &self.viz_pcm);
                        }
                        _ => {}
                    }
                    continue;
                }

                let comp_txt = self.composer.as_text();
                let slash_items = if comp_txt.starts_with('/') && !comp_txt.contains(char::is_whitespace) {
                    let (theme_lbl, _, _) = self.demo.state_flags();
                    let sidebar_on = self.demo.sidebar_visible(tw);
                    let snap = self.snapshot();
                    filter_slash_commands(&comp_txt, &snap, theme_lbl, sidebar_on)
                } else {
                    Vec::new()
                };
                if slash_items.is_empty() {
                    self.slash_sel = 0;
                } else if self.slash_sel >= slash_items.len() {
                    self.slash_sel = slash_items.len() - 1;
                }

                match ev {
                    InputEvent::CtrlQ => {
                        let _ = self.demo.configure("stop_all", "", tw, &self.viz_pcm);
                        self.cancel_active_stream();
                        return Ok(());
                    }
                    InputEvent::CtrlO => {
                        let _ = self.handle_slash_command("/fold", Some(&mut guard));
                    }
                    InputEvent::CtrlB => {
                        let _ = self.handle_slash_command("/inspector", Some(&mut guard));
                    }
                    InputEvent::CtrlT => {
                        let msg = self.demo.configure("theme_cycle", "", tw, &self.viz_pcm);
                        self.set_toast(msg);
                    }
                    InputEvent::CtrlR => {
                        if self.active_recorder.is_some() {
                            self.stop_and_submit_mic();
                        } else if self.asr_rx.is_some() {
                            self.set_toast("Transcribing your voice on RTX 3080...");
                        } else if self.streaming_rx.is_some() {
                            self.set_toast("Wait for current reply or press Esc to interrupt");
                        } else {
                            self.start_mic_recording();
                        }
                    }
                    InputEvent::CtrlC => {
                        if self.active_recorder.is_some()
                            || self.asr_rx.is_some()
                            || self.streaming_rx.is_some()
                            || self.tts_speaking.load(Ordering::Relaxed)
                        {
                            self.cancel_active_stream();
                        } else if !self.composer.is_empty() {
                            self.composer.clear();
                            self.set_toast("Cleared input (press Ctrl+Q or /quit to exit TUI)");
                        } else {
                            return Ok(());
                        }
                    }
                    InputEvent::Esc => {
                        if self.active_recorder.is_some()
                            || self.asr_rx.is_some()
                            || self.hands_free_talk
                            || self.streaming_rx.is_some()
                            || self.tts_speaking.load(Ordering::Relaxed)
                            || self.demo.state_flags().2
                        {
                            self.cancel_active_stream();
                        } else if self.scroll_from_bottom > 0 {
                            self.scroll_from_bottom = 0;
                        } else {
                            self.composer.clear();
                        }
                    }
                    InputEvent::CtrlS => {
                        let _ = self.handle_slash_command("/spec", Some(&mut guard));
                    }
                    InputEvent::CtrlL => {
                        if self.streaming_rx.is_none() {
                            let _ = self.handle_slash_command("/clear", Some(&mut guard));
                        }
                    }
                    InputEvent::CtrlY => {
                        let _ = self.handle_slash_command("/copy", Some(&mut guard));
                    }
                    InputEvent::ScrollUp(n) => {
                        self.scroll_from_bottom = self.scroll_from_bottom.saturating_add(n);
                    }
                    InputEvent::ScrollDown(n) => {
                        self.scroll_from_bottom = self.scroll_from_bottom.saturating_sub(n);
                    }
                    InputEvent::PageUp => {
                        self.scroll_from_bottom = self.scroll_from_bottom.saturating_add(th / 2);
                    }
                    InputEvent::PageDown => {
                        self.scroll_from_bottom = self.scroll_from_bottom.saturating_sub(th / 2);
                    }
                    InputEvent::Tab => {
                        if !slash_items.is_empty() {
                            let chosen = &slash_items[self.slash_sel];
                            let repl = if chosen.args.is_empty() {
                                chosen.cmd.to_string()
                            } else {
                                format!("{} ", chosen.cmd)
                            };
                            self.composer.replace_with(&repl);
                            self.slash_sel = 0;
                        } else {
                            self.composer.insert_str("    ");
                        }
                    }
                    InputEvent::Char(ch) => {
                        self.composer.insert_char(ch);
                        self.slash_sel = 0;
                    }
                    InputEvent::Paste(text) => self.composer.insert_str(&text),
                    InputEvent::Newline => self.composer.insert_char('\n'),
                    InputEvent::Backspace => {
                        self.composer.backspace();
                        self.slash_sel = 0;
                    }
                    InputEvent::Delete | InputEvent::CtrlD => self.composer.delete_forward(),
                    InputEvent::CtrlW => self.composer.delete_word_back(),
                    InputEvent::CtrlU => self.composer.kill_to_start(),
                    InputEvent::Left => self.composer.move_left(),
                    InputEvent::Right => self.composer.move_right(),
                    InputEvent::Home => self.composer.move_home(),
                    InputEvent::End => {
                        if self.composer.is_empty() && self.scroll_from_bottom > 0 {
                            self.scroll_from_bottom = 0;
                        } else {
                            self.composer.move_end();
                        }
                    }
                    InputEvent::Up => {
                        if !slash_items.is_empty() {
                            self.slash_sel = if self.slash_sel == 0 {
                                slash_items.len() - 1
                            } else {
                                self.slash_sel - 1
                            };
                        } else {
                            self.composer.move_vertical(-1);
                        }
                    }
                    InputEvent::Down => {
                        if !slash_items.is_empty() {
                            self.slash_sel = (self.slash_sel + 1) % slash_items.len();
                        } else {
                            self.composer.move_vertical(1);
                        }
                    }
                    InputEvent::Enter => {
                        if self.active_recorder.is_some() {
                            self.stop_and_submit_mic();
                            continue;
                        }
                        if self.asr_rx.is_some() || self.streaming_rx.is_some() {
                            self.set_toast("Wait for current voice/text turn or press Esc to stop first");
                            continue;
                        }
                        if !slash_items.is_empty() {
                            let chosen = &slash_items[self.slash_sel];
                            if chosen.args.starts_with('<')
                                && comp_txt.trim() != chosen.cmd
                                && chosen.cmd != "/effort"
                            {
                                self.composer.replace_with(&format!("{} ", chosen.cmd));
                                self.slash_sel = 0;
                                continue;
                            }
                            let selected_cmd = chosen.cmd.to_string();
                            let _ = self.composer.take_submission();
                            self.slash_sel = 0;
                            if let Some(should_quit) = self.handle_slash_command(&selected_cmd, Some(&mut guard)) {
                                if should_quit {
                                    return Ok(());
                                }
                            }
                            continue;
                        }
                        let Some(trimmed) = self.composer.take_submission() else {
                            continue;
                        };
                        if let Some(should_quit) = self.handle_slash_command(&trimmed, Some(&mut guard)) {
                            if should_quit {
                                return Ok(());
                            }
                            continue;
                        }
                        self.start_generation(trimmed, false);
                    }
                }
            }
        }
    }

    fn drain_stream(&mut self, timeout_s: u64, label: &str) -> Result<(), String> {
        let t0 = Instant::now();
        while self.streaming_rx.is_some() {
            self.pump_stream();
            if t0.elapsed() > Duration::from_secs(timeout_s) {
                return Err(format!("{} timed out", label));
            }
            thread::sleep(Duration::from_millis(4));
        }
        Ok(())
    }

    fn run_self_test(model_ref: &str, max_ctx: usize, spec: bool, temp: f32) -> Result<(), String> {
        let mut app = Self::new(model_ref, max_ctx, spec, temp, 128)?;
        let mut scr = ScreenBuf::new(140, 36);
        let _ = app.handle_slash_command("/verify", None);
        let _ = app.draw(&mut scr, 140, 36);

        app.start_generation(
            "Remember the secret code `RTX3080-GA102`. Reply in one short sentence confirming it.".into(),
            true,
        );
        app.drain_stream(15, "TUI self-test turn 1")?;
        let reply1 = app.messages.last().map(|m| m.content.clone()).unwrap_or_default();

        app.start_generation("What was the exact secret code I just gave you?".into(), false);
        app.drain_stream(15, "TUI self-test turn 2")?;
        let reply2 = app.messages.last().map(|m| m.content.clone()).unwrap_or_default();

        let cursor = app.draw(&mut scr, 140, 36);
        let ansi_bytes = scr.render_ansi(cursor).len();
        println!(
            "TUI_SELF_TEST_OK | model='{}' | ping={:.2}ms | turn1_tps={:.2} | turn2_cached_prefix={} toks | ansi_frame={}B | reply1={:?} | reply2={:?}",
            app.cfg.model_name, app.ping_ms, app.last_tps, app.last_cached_prefix, ansi_bytes, reply1.trim(), reply2.trim()
        );
        Ok(())
    }

    fn run_voice_self_test(model_ref: &str, max_ctx: usize, spec: bool, temp: f32) -> Result<(), String> {
        let mut app = Self::new(model_ref, max_ctx, spec, temp, 96)?;
        let voice_ref = app.asr_model_ref.clone();
        let _ = DaemonControl::query(DaemonQuery::AsrWarm(voice_ref.clone()))?;
        let _ = DaemonControl::query(DaemonQuery::TtsWarm(voice_ref.clone()))?;
        let pcm_16k = prepare_voice_self_test_pcm(&voice_ref)?;
        let mut scr = ScreenBuf::new(140, 36);

        app.submit_pcm_for_transcription(pcm_16k);
        let t_stt = Instant::now();
        while app.asr_rx.is_some() {
            app.pump_voice_input();
            app.voice_enabled = false;
            if t_stt.elapsed() > Duration::from_secs(20) {
                return Err("TUI voice self-test STT timed out".into());
            }
            thread::sleep(Duration::from_millis(3));
        }
        let stt_ms = app.last_asr_ms;
        let user_msg = app.messages.iter().find(|m| m.role == "user").map(|m| m.content.clone())
            .ok_or("Expected transcribed user message in TUI chat window")?;

        let t_llm = Instant::now();
        app.drain_stream(20, "TUI voice self-test LLM generation")?;
        let llm_ms = t_llm.elapsed().as_secs_f64() * 1000.0;
        let asst_msg = app.messages.iter().rev().find(|m| m.role == "assistant").map(|m| m.content.clone())
            .ok_or("Expected assistant reply in TUI chat window")?;

        let (reply_pcm, ttfa_ms) = measure_voice_self_test_tts(&voice_ref, &asst_msg)?;
        let cursor = app.draw(&mut scr, 140, 36);
        let ansi_bytes = scr.render_ansi(cursor).len();
        println!(
            "TUI_VOICE_SELF_TEST_OK | stt_ms={:.2} | llm_tps={:.2} | llm_ms={:.2} | tts_ttfa_ms={:.2} | speech_to_voice_ms={:.2} | reply_audio_s={:.2} | chat_msgs={} | ansi_frame={}B | transcript={:?} | reply_preview={:?}",
            stt_ms, app.last_tps, llm_ms, ttfa_ms, stt_ms + llm_ms + ttfa_ms,
            reply_pcm.len() as f64 / (TTS_SAMPLE_RATE as f64), app.messages.len(), ansi_bytes,
            user_msg, asst_msg.chars().take(90).collect::<String>()
        );
        Ok(())
    }
}

pub fn run_tui(
    model_ref: &str, max_ctx: usize, spec: bool, temp: f32, max_tokens: usize,
    self_test: bool, voice_start: bool, voice_self_test: bool,
) -> Result<(), String> {
    if voice_self_test {
        return TuiApp::run_voice_self_test(model_ref, max_ctx, spec, temp);
    }
    if self_test {
        return TuiApp::run_self_test(model_ref, max_ctx, spec, temp);
    }
    TuiApp::new(model_ref, max_ctx, spec, temp, max_tokens)?.run_interactive_loop(voice_start)
}

#[cfg(test)]
mod tests;

