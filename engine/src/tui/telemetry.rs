//! TUI Daemon Stream Workers, Telemetry JSON Parser, Status Badges, and Voice Playback Helpers.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use super::markdown::{sanitize_text_for_speech, split_spoken_sentences};
use crate::runtime::client::{DaemonControl, DaemonQuery, DaemonStream};
use crate::topic_lora::json_find_num;
use crate::voice::audio::{
    read_wav_mono_16k, resample_sinc_mono, CoreAudioPlayer, ASR_SAMPLE_RATE, TTS_SAMPLE_RATE,
};
use crate::voice::tts::TtsSynthParams;

#[derive(Clone, Default)]
pub(super) struct TurnStats {
    pub decode_tok_s: f64,
    pub ttft_ms: f64,
    pub completion_tokens: usize,
    pub prompt_tokens: usize,
    pub cached_prefix_tokens: usize,
    pub kernel_launches: usize,
    pub spec_accepted_extra: usize,
    pub token_ids: Vec<i32>,
    pub cancelled: bool,
}

pub(super) fn parse_stats_json(j: &str) -> TurnStats {
    if j.contains("\"cancelled\":true") {
        return TurnStats {
            cancelled: true,
            ..Default::default()
        };
    }
    let get_num = |key: &str| -> f64 {
        j.find(key)
            .and_then(|idx| {
                let rest = j[idx + key.len()..].trim_start().strip_prefix(':')?.trim_start();
                let s: String = rest
                    .chars()
                    .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
                    .collect();
                s.parse::<f64>().ok()
            })
            .unwrap_or(0.0)
    };
    let token_ids = j
        .find("\"token_ids\":[")
        .and_then(|idx| {
            let start = idx + 13;
            let end = j[start..].find(']')? + start;
            Some(
                j[start..end]
                    .split(',')
                    .filter_map(|p| p.trim().parse::<i32>().ok())
                    .collect::<Vec<i32>>(),
            )
        })
        .unwrap_or_default();
    TurnStats {
        decode_tok_s: get_num("\"decode_tok_per_s\""),
        ttft_ms: get_num("\"ttft_ms\""),
        completion_tokens: get_num("\"completion_tokens\"") as usize,
        prompt_tokens: get_num("\"prompt_tokens\"") as usize,
        cached_prefix_tokens: get_num("\"cached_prefix_tokens\"") as usize,
        kernel_launches: get_num("\"kernel_launches\"") as usize,
        spec_accepted_extra: get_num("\"spec_accepted_extra\"") as usize,
        token_ids,
        cancelled: false,
    }
}

pub(super) enum StreamEvent {
    Token {
        piece: String,
        tok_count: usize,
        elapsed_s: f64,
        ttft_ms: f64,
    },
    Finished(TurnStats),
    Error(String),
}

pub(super) fn spawn_generation_stream(
    prompt_ids: Vec<i32>,
    max_toks: usize,
    temp: f32,
    spec: bool,
    force_reset_cache: bool,
) -> (Receiver<StreamEvent>, Arc<AtomicBool>) {
    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_bg = Arc::clone(&cancel);
    let (tx, rx) = mpsc::channel::<StreamEvent>();

    thread::spawn(move || {
        let t_req = Instant::now();
        let mut t_first: Option<Instant> = None;
        let mut ttft_ms = 0.0f64;
        let mut count = 0usize;
        let mut streamed_ids = Vec::with_capacity(256);

        let res = DaemonStream::generate_ids_cancellable(
            &prompt_ids,
            max_toks,
            temp,
            spec,
            force_reset_cache,
            &cancel_bg,
            |tok_id, piece| {
                if tok_id >= 0 {
                    streamed_ids.push(tok_id);
                }
                count += 1;
                let now = Instant::now();
                if t_first.is_none() {
                    t_first = Some(now);
                    ttft_ms = t_req.elapsed().as_secs_f64() * 1000.0;
                }
                let dec_elapsed = t_first
                    .map(|t0| now.duration_since(t0).as_secs_f64())
                    .unwrap_or(0.0);
                if !piece.is_empty() {
                    let _ = tx.send(StreamEvent::Token {
                        piece: piece.to_string(),
                        tok_count: count,
                        elapsed_s: dec_elapsed,
                        ttft_ms,
                    });
                }
            },
        );

        match res {
            Ok(stats_json) => {
                let mut st = parse_stats_json(&stats_json);
                if st.cancelled {
                    st.completion_tokens = count;
                    st.ttft_ms = ttft_ms;
                    st.token_ids = streamed_ids;
                }
                let _ = tx.send(StreamEvent::Finished(st));
            }
            Err(e) => {
                let _ = tx.send(StreamEvent::Error(e));
            }
        }
    });

    (rx, cancel)
}

pub(super) fn spawn_tts_sentences(
    sentences: Vec<String>,
    tts_ref: String,
    speaker: String,
    lang: String,
    speaking_bg: Arc<AtomicBool>,
    ttfa_bg: Arc<Mutex<f64>>,
) -> Arc<AtomicBool> {
    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_bg = Arc::clone(&cancel);
    speaking_bg.store(true, Ordering::Relaxed);

    thread::spawn(move || {
        let Ok(mut player) = CoreAudioPlayer::new(TTS_SAMPLE_RATE) else {
            speaking_bg.store(false, Ordering::Relaxed);
            return;
        };
        let t_turn = Instant::now();
        let mut recorded_ttfa = false;

        for sent in sentences {
            if cancel_bg.load(Ordering::Relaxed) {
                player.abort_immediate();
                speaking_bg.store(false, Ordering::Relaxed);
                return;
            }
            let params = TtsSynthParams {
                text: sent,
                speaker: speaker.clone(),
                language: lang.clone(),
                streaming_chunks: true,
                ..TtsSynthParams::default()
            };
            let _ = DaemonStream::speak(&tts_ref, &params, Some(&cancel_bg), |chunk| {
                if cancel_bg.load(Ordering::Relaxed) {
                    player.abort_immediate();
                    return false;
                }
                if !recorded_ttfa {
                    recorded_ttfa = true;
                    if let Ok(mut g) = ttfa_bg.lock() {
                        *g = t_turn.elapsed().as_secs_f64() * 1000.0;
                    }
                }
                let _ = player.push_samples(chunk);
                true
            });
        }

        if cancel_bg.load(Ordering::Relaxed) {
            player.abort_immediate();
        } else {
            player.drain();
        }
        speaking_bg.store(false, Ordering::Relaxed);
    });

    cancel
}

pub(super) fn prepare_voice_self_test_pcm(voice_ref: &str) -> Result<Vec<f32>, String> {
    let wav_path = std::path::Path::new("/tmp/spoken_question_16k.wav");
    if wav_path.exists() {
        read_wav_mono_16k(wav_path)
    } else {
        let q_params = TtsSynthParams {
            text: "What makes GDDR6X memory bandwidth so fast for running AI models locally?".into(),
            speaker: "ryan".into(),
            language: "english".into(),
            seed: Some(42),
            streaming_chunks: false,
            ..Default::default()
        };
        let (wav_24k, _) = DaemonStream::speak(voice_ref, &q_params, None, |_| true)?;
        Ok(resample_sinc_mono(&wav_24k, TTS_SAMPLE_RATE, ASR_SAMPLE_RATE))
    }
}

pub(super) fn measure_voice_self_test_tts(
    voice_ref: &str,
    asst_msg: &str,
) -> Result<(Vec<f32>, f64), String> {
    let clean_reply = sanitize_text_for_speech(asst_msg);
    let first_sent = split_spoken_sentences(&clean_reply)
        .into_iter()
        .next()
        .unwrap_or(clean_reply);
    let tts_params = TtsSynthParams {
        text: first_sent,
        speaker: "ryan".into(),
        language: "english".into(),
        seed: Some(7),
        streaming_chunks: true,
        ..Default::default()
    };
    let t_tts = Instant::now();
    let mut ttfa_ms = 0.0f64;
    let mut got_first_audio = false;
    let (reply_pcm, _) = DaemonStream::speak(voice_ref, &tts_params, None, |chunk| {
        if !got_first_audio && !chunk.is_empty() {
            got_first_audio = true;
            ttfa_ms = t_tts.elapsed().as_secs_f64() * 1000.0;
        }
        true
    })?;
    Ok((reply_pcm, ttfa_ms))
}

pub(super) fn query_lora_badge_from_daemon() -> Option<String> {
    let Ok(j) = DaemonControl::query(DaemonQuery::LoraStatus) else {
        return None;
    };
    if !j.contains("\"active\":true") {
        return None;
    }
    let rank = j
        .find("\"rank\":")
        .and_then(|idx| {
            let s: String = j[idx + 7..]
                .chars()
                .skip_while(|c| c.is_whitespace())
                .take_while(|c| c.is_ascii_digit())
                .collect();
            s.parse::<usize>().ok()
        })
        .unwrap_or(0);
    Some(format!(" LoRA r={} ", rank))
}

pub(super) fn query_rag_badge_from_daemon() -> Option<String> {
    let Ok(j) = DaemonControl::query(DaemonQuery::RagStatus) else {
        return None;
    };
    if !j.contains("\"active\":true") {
        return None;
    }
    let ndocs = j
        .find("\"num_docs\":")
        .and_then(|idx| {
            let s: String = j[idx + 11..]
                .chars()
                .skip_while(|c| c.is_whitespace())
                .take_while(|c| c.is_ascii_digit())
                .collect();
            s.parse::<usize>().ok()
        })
        .unwrap_or(0);
    let strat = j
        .find("\"strategy\":\"")
        .and_then(|idx| {
            let rest = &j[idx + 12..];
            rest.find('"').map(|e| &rest[..e])
        })
        .unwrap_or("rag");
    Some(format!(" 📚 RAG:{} ({}) ", ndocs, strat))
}

pub(super) fn query_topic_badge_from_daemon() -> Option<String> {
    let Ok(j) = DaemonControl::query(DaemonQuery::TopicStatus) else {
        return None;
    };
    if !j.contains("\"active\":true") {
        return None;
    };
    let ntopics = json_find_num(&j, "num_topics").unwrap_or(0.0) as usize;
    Some(format!(" ⚡ TOPIC:{} ", ntopics))
}
