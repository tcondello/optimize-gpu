mod asr;
mod audio;
mod daemon;
mod engine;
mod gguf;
mod lora;
mod nv_defs;
mod nv_hw;
mod tinygpu_client;
mod tts;
mod tui;
mod unicode_cats;

use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::thread;
use std::time::{Duration, Instant};

use audio::{
    read_wav_mono_16k, write_wav_pcm16, CoreAudioPlayer, CoreAudioRecorder, ASR_SAMPLE_RATE,
    TTS_SAMPLE_RATE,
};
use daemon::{
    client_asr_warm, client_codec_decode_via_daemon, client_generate_via_daemon, client_lora_clear,
    client_lora_load, client_lora_status, client_speak_via_daemon, client_transcribe_via_daemon,
    client_tts_warm, daemon_ping, ensure_daemon_running, format_stats_json,
    install_signal_handlers, run_daemon_foreground, run_http_server, stop_daemon,
};
use engine::QwenMegakernelEngine;
use gguf::{resolve_model_path, BpeTokenizer, GgufFile, QwenModelConfig, Utf8StreamDecoder};
use lora::write_synthetic_peft_adapter;
use tts::TtsSynthParams;

fn print_usage() {
    eprintln!(
        "qwen-nv v0.4.0 — Zero-Docker, Zero-Crate Native ARM64 Rust + GA102 (sm_86) Modular Text + STT + TTS + LoRA Engine\n\n\
         USAGE:\n  \
           qwen-nv tui        [OPTIONS]                Launch interactive TrueColor Rust TUI (`Ctrl+R` mic, `/talk`, `/speak`, `/lora`)\n  \
           qwen-nv run        [OPTIONS] <PROMPT>       Generate text response (uses persistent <1ms daemon by default)\n  \
           qwen-nv transcribe [OPTIONS] <WAV|--mic>    100%-on-RTX-3080 Qwen3-ASR-1.7B Q8_0 speech-to-text (alias: listen, stt)\n  \
           qwen-nv speak      [OPTIONS] <TEXT>         100%-on-RTX-3080 Qwen3-TTS-1.7B Q8_0 streaming speech synthesis\n  \
           qwen-nv lora       <load|clear|status|...>  Dynamic <5ms Rank-r (r<=32) LoRA hot-swap on base Qwen Thinker\n  \
           qwen-nv asr-warm   [OPTIONS]                Pre-load co-resident Qwen3-ASR-1.7B Q8_0 weights into VRAM\n  \
           qwen-nv tts-warm   [OPTIONS]                Pre-load co-resident Qwen3-TTS-1.7B + Code2Wav weights into VRAM\n  \
           qwen-nv codec-decode <CODES_BIN> <OUT_PATH> Run 100%-GPU 12Hz Code2Wav streaming vocoder on [T, 16] i32 codes\n  \
           qwen-nv daemon     [OPTIONS]                Run persistent GPU daemon (~/.cache/qwen-nv-rs/daemon.sock)\n  \
           qwen-nv status                              Query running GPU daemon status & warm ping latency\n  \
           qwen-nv stop                                Cleanly unload GSP-RM and stop running GPU daemon\n  \
           qwen-nv serve      [OPTIONS]                Serve OpenAI HTTP API (/v1/chat/completions, /v1/audio/speech, /v1/audio/transcriptions)\n  \
           qwen-nv tokenize   [OPTIONS] <TEXT>         Offline GGUF BPE tokenizer check (zero GPU required)\n\n\
         TEXT & DAEMON OPTIONS:\n  \
           --model <REF>        Thinker GGUF alias (qwen3:4b, qwen2.5:3b, ...) or path [default: qwen3:4b]\n  \
           --max-tokens <N>     Maximum tokens to generate [default: 256]\n  \
           --temp <F>           Sampling temperature (0.0 = greedy for text; 0.9 default for speak)\n  \
           --ctx <N>            Maximum KV context length [default: 4096]\n  \
           --spec / --no-spec   Enable/disable lossless BM=4 N-gram speculative verify [default: --spec]\n  \
           --direct             Run standalone in-process instead of via persistent daemon\n  \
           --background         Spawn daemon in background and return once ready\n  \
           --reset-cache        Clear KV prefix cache before running prompt\n  \
           --raw                Encode raw text instead of ChatML (<|im_start|>user...)\n  \
           --quiet              Suppress streamed stdout text\n  \
           --json-stats         Print single-line JSON stats summary\n  \
           --port <PORT>        HTTP server port for `serve` [default: 8000]\n\n\
         STT (`transcribe` / `listen`) OPTIONS:\n  \
           --asr-model <REF>    ASR model alias (1.7b [Q8_0 default], 1.7b-q6, 1.7b-q4, 0.6b) or GGUF path [default: 1.7b]\n  \
           --mic                Record live 16 kHz audio from macOS microphone (auto-VAD stops on ~1s pause)\n  \
           --duration <SEC>     Fixed microphone recording duration in seconds (optional; overrides auto-VAD)\n  \
           --ask                Full round-trip voice assistant: Transcribe -> Qwen3-4B -> Qwen3-TTS-1.7B voice reply!\n\n\
         TTS (`speak`) OPTIONS:\n  \
           --tts-model <REF>    TTS model alias (1.7b [Q8_0 default], 1.7b-q4, 0.6b) or GGUF path [default: 1.7b]\n  \
           --voice <NAME>       Speaker preset: ryan, serena, vivian, aiden, uncle_fu, ono_anna, sohee, eric, dylan [default: ryan]\n  \
           --lang <LANG>        Language tag: english, auto, chinese, japanese, korean, french, german, spanish [default: english]\n  \
           --instruct <STR>     Optional voice style instruction (CustomVoice 1.7B)\n  \
           -o, --output <PATH>  Save synthesized 24 kHz audio to .wav (16-bit PCM) or .bin (raw f32)\n  \
           --no-play            Skip live CoreAudio speaker playback (useful for batch WAV rendering & benchmarks)\n  \
           --seed <U64>         Deterministic Philox4x32-10 RNG seed\n  \
           --greedy             Greedy argmax codec sampling (with repetition_penalty=1.05)\n  \
           --max-frames <N>     Max 12.5 Hz codec frames (80 ms/frame) [default: 1024]"
    );
}

fn write_audio_output(path: &Path, samples: &[f32]) -> Result<(), String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if ext == "bin" || ext == "f32" || ext == "raw" {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
        }
        let mut raw = Vec::with_capacity(samples.len() * 4);
        for &s in samples {
            raw.extend_from_slice(&s.to_bits().to_le_bytes());
        }
        fs::write(path, raw).map_err(|e| format!("Failed writing raw f32 {}: {}", path.display(), e))
    } else {
        write_wav_pcm16(path, samples, TTS_SAMPLE_RATE)
    }
}

fn main() {
    if let Err(e) = run_cli() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run_cli() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        print_usage();
        return Ok(());
    }

    let subcmd = args[0].as_str();
    let mut model_ref = "qwen3:4b".to_string();
    let mut tts_model_ref = "1.7b".to_string();
    let mut asr_model_ref = "1.7b".to_string();
    let mut max_tokens = 256usize;
    let mut max_tokens_explicit = false;
    let mut temp = 0.0f32;
    let mut temp_explicit = false;
    let mut max_ctx = 4096usize;
    let mut spec = true;
    let mut direct = false;
    let mut background = false;
    let mut reset_cache = false;
    let mut raw_mode = false;
    let mut quiet = false;
    let mut json_stats = false;
    let mut self_test = false;
    let mut voice_self_test = false;
    let mut voice_chat_start = false;
    let mut port = 8000u16;

    // STT-specific options
    let mut use_mic = false;
    let mut mic_duration: Option<f64> = None;
    let mut ask_after_stt = false;
    let mut lang_explicit = false;

    // TTS-specific options
    let mut voice = "ryan".to_string();
    let mut lang = "english".to_string();
    let mut instruct = String::new();
    let mut output_path: Option<String> = None;
    let mut no_play = false;
    let mut greedy = false;
    let mut seed: Option<u64> = None;
    let mut top_k = 50i32;
    let mut rep_pen = 1.05f32;
    let mut max_frames = 1024usize;
    let mut stream_chunks = true;

    // LoRA options
    let mut lora_scale = 1.0f32;
    let mut lora_rank = 16usize;
    let mut lora_alpha = 32.0f32;
    let mut lora_strength = 0.25f32;

    let mut positional = Vec::<String>::new();

    let start_idx = if matches!(
        subcmd,
        "run"
            | "tui"
            | "chat"
            | "speak"
            | "transcribe"
            | "listen"
            | "stt"
            | "lora"
            | "asr-warm"
            | "tts-warm"
            | "codec-decode"
            | "daemon"
            | "status"
            | "stop"
            | "serve"
            | "tokenize"
    ) {
        1
    } else {
        0
    };
    let effective_cmd = if start_idx == 1 { subcmd } else { "run" };

    let mut i = start_idx;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "-m" => {
                i += 1;
                if i < args.len() {
                    model_ref = args[i].clone();
                }
            }
            "--tts-model" => {
                i += 1;
                if i < args.len() {
                    tts_model_ref = args[i].clone();
                }
            }
            "--asr-model" => {
                i += 1;
                if i < args.len() {
                    asr_model_ref = args[i].clone();
                }
            }
            "--duration" | "-d" => {
                i += 1;
                if i < args.len() {
                    mic_duration = args[i].parse::<f64>().ok();
                }
            }
            "--max-tokens" | "-n" => {
                i += 1;
                if i < args.len() {
                    max_tokens = args[i].parse().unwrap_or(256);
                    max_tokens_explicit = true;
                }
            }
            "--temp" | "-t" => {
                i += 1;
                if i < args.len() {
                    temp = args[i].parse().unwrap_or(0.0);
                    temp_explicit = true;
                }
            }
            "--ctx" => {
                i += 1;
                if i < args.len() {
                    max_ctx = args[i].parse().unwrap_or(4096);
                }
            }
            "--port" => {
                i += 1;
                if i < args.len() {
                    port = args[i].parse().unwrap_or(8000);
                }
            }
            "--voice" | "--speaker" | "-s" => {
                i += 1;
                if i < args.len() {
                    voice = args[i].clone();
                }
            }
            "--lang" | "--language" | "-l" => {
                i += 1;
                if i < args.len() {
                    lang = args[i].clone();
                    lang_explicit = true;
                }
            }
            "--instruct" => {
                i += 1;
                if i < args.len() {
                    instruct = args[i].clone();
                }
            }
            "-o" | "--output" => {
                i += 1;
                if i < args.len() {
                    output_path = Some(args[i].clone());
                }
            }
            "--seed" => {
                i += 1;
                if i < args.len() {
                    seed = args[i].parse::<u64>().ok();
                }
            }
            "--top-k" => {
                i += 1;
                if i < args.len() {
                    top_k = args[i].parse().unwrap_or(50);
                }
            }
            "--rep-pen" | "--repetition-penalty" => {
                i += 1;
                if i < args.len() {
                    rep_pen = args[i].parse().unwrap_or(1.05);
                }
            }
            "--max-frames" | "--max-steps" => {
                i += 1;
                if i < args.len() {
                    max_frames = args[i].parse().unwrap_or(1024);
                }
            }
            "--scale" => {
                i += 1;
                if i < args.len() {
                    lora_scale = args[i].parse().unwrap_or(1.0);
                }
            }
            "--rank" => {
                i += 1;
                if i < args.len() {
                    lora_rank = args[i].parse().unwrap_or(16);
                }
            }
            "--alpha" => {
                i += 1;
                if i < args.len() {
                    lora_alpha = args[i].parse().unwrap_or(32.0);
                }
            }
            "--strength" => {
                i += 1;
                if i < args.len() {
                    lora_strength = args[i].parse().unwrap_or(0.25);
                }
            }
            "--no-play" => no_play = true,
            "--play" => no_play = false,
            "--greedy" => greedy = true,
            "--no-stream-chunks" => stream_chunks = false,
            "--spec" => spec = true,
            "--no-spec" => spec = false,
            "--direct" => direct = true,
            "--background" => background = true,
            "--reset-cache" => reset_cache = true,
            "--raw" => raw_mode = true,
            "--quiet" | "-q" => quiet = true,
            "--json-stats" => json_stats = true,
            "--self-test" => self_test = true,
            "--voice-self-test" => voice_self_test = true,
            "--talk" | "--voice-chat" => voice_chat_start = true,
            "--mic" => use_mic = true,
            "--ask" => ask_after_stt = true,
            "-p" | "--prompt" | "--text" => {
                i += 1;
                if i < args.len() {
                    positional.push(args[i].clone());
                }
            }
            other => positional.push(other.to_string()),
        }
        i += 1;
    }

    match effective_cmd {
        "status" => {
            let t0 = Instant::now();
            match daemon_ping() {
                Ok((mref, mname, sms)) => {
                    let us = t0.elapsed().as_secs_f64() * 1e6;
                    let lora_info = client_lora_status().unwrap_or_default();
                    println!(
                        "ONLINE | model_ref={} | model_name='{}' | active_SMs={} | ping={:.1} us | lora={}",
                        mref, mname, sms, us, lora_info
                    );
                    Ok(())
                }
                Err(_) => {
                    println!("OFFLINE");
                    Ok(())
                }
            }
        }
        "stop" => {
            let stopped = stop_daemon()?;
            if stopped {
                eprintln!("[qwen-nv] Daemon stopped and GSP-RM unloaded cleanly.");
            } else {
                eprintln!("[qwen-nv] No running daemon found.");
            }
            Ok(())
        }
        "daemon" => {
            if background {
                ensure_daemon_running(&model_ref, max_ctx)
            } else {
                run_daemon_foreground(&model_ref, max_ctx)
            }
        }
        "serve" => run_http_server(&model_ref, max_ctx, port, spec),
        "tui" | "chat" => {
            let tui_max = if max_tokens_explicit { max_tokens } else { 1024 };
            tui::run_tui(
                &model_ref,
                max_ctx,
                spec,
                temp,
                tui_max,
                self_test,
                voice_chat_start,
                voice_self_test,
            )
        }
        "asr-warm" => {
            ensure_daemon_running(&model_ref, max_ctx)?;
            let res = client_asr_warm(&asr_model_ref)?;
            println!("{}", res);
            Ok(())
        }
        "tts-warm" => {
            ensure_daemon_running(&model_ref, max_ctx)?;
            let res = client_tts_warm(&tts_model_ref)?;
            println!("{}", res);
            Ok(())
        }
        "transcribe" | "listen" | "stt" => {
            let asr_lang = if lang_explicit { lang.as_str() } else { "auto" };

            // Ensure daemon + ASR weights are ready BEFORE opening live mic so user speaks with zero delay
            if !direct {
                ensure_daemon_running(&model_ref, max_ctx)?;
                if use_mic {
                    let _ = client_asr_warm(&asr_model_ref)?;
                    if ask_after_stt && !no_play {
                        let _ = client_tts_warm(&tts_model_ref)?;
                    }
                }
            }

            let pcm_16k = if use_mic {
                let mut rec = CoreAudioRecorder::start(ASR_SAMPLE_RATE)?;
                if let Some(sec) = mic_duration {
                    eprintln!("[qwen-nv listen] 🎙 Recording microphone for {:.1}s (@ 16 kHz)...", sec);
                    thread::sleep(Duration::from_secs_f64(sec.max(0.2)));
                } else {
                    eprintln!("[qwen-nv listen] 🎙 Speak your question now (auto-stops ~1s after you pause)...");
                    let t_rec = Instant::now();
                    loop {
                        let st = rec.live_stats();
                        if st.auto_stop_ready || t_rec.elapsed() > Duration::from_secs(30) {
                            break;
                        }
                        thread::sleep(Duration::from_millis(25));
                    }
                }
                let samples = rec.finish_and_take_samples();
                if let Some(out) = &output_path {
                    write_wav_pcm16(Path::new(out), &samples, ASR_SAMPLE_RATE)?;
                }
                samples
            } else {
                let wav_arg = positional.first().ok_or_else(|| {
                    "Usage: qwen-nv transcribe <AUDIO.wav> [--asr-model 1.7b] (or pass --mic)".to_string()
                })?;
                read_wav_mono_16k(Path::new(wav_arg))?
            };

            if direct {
                install_signal_handlers();
                let mut engine = QwenMegakernelEngine::load(&model_ref, max_ctx)?;
                let lang_opt = if asr_lang.eq_ignore_ascii_case("auto") {
                    None
                } else {
                    Some(asr_lang)
                };
                let stats = engine.transcribe_audio(&asr_model_ref, &pcm_16k, lang_opt)?;
                if json_stats {
                    println!("{}", stats.to_json());
                } else {
                    println!("{}", stats.text);
                    if !quiet {
                        eprintln!(
                            "[qwen-nv transcribe] {:.2}s audio -> {:.1} ms ({:.1}x real-time, RTF {:.4}, lang={})",
                            stats.audio_sec,
                            stats.total_ms,
                            stats.speed_x_realtime,
                            stats.rtf,
                            stats.detected_language
                        );
                    }
                }
                engine.gpu.shutdown_clean();
                Ok(())
            } else {
                let t_ipc = Instant::now();
                let (transcript, stats_json) =
                    client_transcribe_via_daemon(&asr_model_ref, asr_lang, &pcm_16k)?;
                let client_wall_ms = t_ipc.elapsed().as_secs_f64() * 1000.0;

                if json_stats && !ask_after_stt {
                    if let Some(stripped) = stats_json.strip_suffix('}') {
                        println!("{},\"client_wall_ms\":{:.3}}}", stripped, client_wall_ms);
                    } else {
                        println!("{}", stats_json);
                    }
                    return Ok(());
                }

                if !ask_after_stt {
                    println!("{}", transcript);
                    if !quiet {
                        eprintln!(
                            "[qwen-nv transcribe] client_wall={:.1} ms | {}",
                            client_wall_ms, stats_json
                        );
                    }
                    return Ok(());
                }

                // Full Round-Trip Voice Assistant (`--ask`): STT -> Qwen3-4B -> Qwen3-TTS-1.7B!
                println!("▸ YOU (RTX 3080 STT {:.1} ms): {}", client_wall_ms, transcript);
                if transcript.trim().is_empty() {
                    return Ok(());
                }
                print!("◆ Qwen3-4B: ");
                let _ = io::stdout().flush();
                let mut reply_text = String::new();
                let llm_stats = client_generate_via_daemon(
                    &transcript,
                    "chat",
                    max_tokens,
                    temp,
                    spec,
                    reset_cache,
                    |_, piece| {
                        reply_text.push_str(piece);
                        print!("{}", piece);
                        let _ = io::stdout().flush();
                    },
                )?;
                println!();

                let mut player = if !no_play {
                    Some(CoreAudioPlayer::new(TTS_SAMPLE_RATE)?)
                } else {
                    None
                };
                let tts_params = TtsSynthParams {
                    text: reply_text,
                    speaker: voice,
                    language: "english".into(),
                    streaming_chunks: true,
                    ..Default::default()
                };
                let t_tts = Instant::now();
                let mut ttfa_ms = 0.0f64;
                let mut got_first = false;
                let (_, tts_stats) =
                    client_speak_via_daemon(&tts_model_ref, &tts_params, None, |chunk| {
                        if !got_first && !chunk.is_empty() {
                            got_first = true;
                            ttfa_ms = t_tts.elapsed().as_secs_f64() * 1000.0;
                        }
                        if let Some(p) = player.as_mut() {
                            let _ = p.push_samples(chunk);
                        }
                        true
                    })?;
                if let Some(mut p) = player {
                    p.drain();
                }
                if !quiet {
                    eprintln!(
                        "[qwen-nv voice-loop] STT={:.1} ms | TTS_TTFA={:.1} ms | LLM={} | TTS={}",
                        client_wall_ms, ttfa_ms, llm_stats, tts_stats
                    );
                }
                Ok(())
            }
        }
        "lora" => {
            let action = positional.first().map(|s| s.as_str()).unwrap_or("status");
            match action {
                "synth-test" => {
                    let out_dir = positional
                        .get(1)
                        .ok_or_else(|| "Usage: qwen-nv lora synth-test <OUT_DIR> [--rank 16] [--strength 0.25]".to_string())?;
                    let path = resolve_model_path(&model_ref)?;
                    let gguf = GgufFile::open(&path)?;
                    let tok = BpeTokenizer::from_gguf(&gguf)?;
                    let cfg = QwenModelConfig::from_gguf(&gguf, &tok, max_ctx, &model_ref)?;
                    let q_dim = cfg.n_heads * cfg.head_dim;
                    let kv_dim = cfg.n_kv_heads * cfg.head_dim;
                    write_synthetic_peft_adapter(
                        Path::new(out_dir),
                        cfg.num_layers,
                        cfg.dim,
                        q_dim,
                        kv_dim,
                        lora_rank,
                        lora_alpha,
                        lora_strength,
                    )?;
                    println!(
                        "{{\"status\":\"ok\",\"out_dir\":\"{}\",\"layers\":{},\"rank\":{},\"alpha\":{:.1},\"strength\":{:.3}}}",
                        out_dir, cfg.num_layers, lora_rank, lora_alpha, lora_strength
                    );
                    Ok(())
                }
                "load" => {
                    let adapter_path = positional
                        .get(1)
                        .ok_or_else(|| "Usage: qwen-nv lora load <ADAPTER_DIR_OR_SAFETENSORS> [--scale 1.0]".to_string())?;
                    ensure_daemon_running(&model_ref, max_ctx)?;
                    let t0 = Instant::now();
                    let res = client_lora_load(adapter_path, lora_scale)?;
                    let client_ms = t0.elapsed().as_secs_f64() * 1000.0;
                    if let Some(stripped) = res.strip_suffix('}') {
                        println!("{},\"client_ipc_ms\":{:.3}}}", stripped, client_ms);
                    } else {
                        println!("{}", res);
                    }
                    Ok(())
                }
                "clear" | "off" | "unload" => {
                    ensure_daemon_running(&model_ref, max_ctx)?;
                    let t0 = Instant::now();
                    let res = client_lora_clear()?;
                    let client_ms = t0.elapsed().as_secs_f64() * 1000.0;
                    if let Some(stripped) = res.strip_suffix('}') {
                        println!("{},\"client_ipc_ms\":{:.3}}}", stripped, client_ms);
                    } else {
                        println!("{}", res);
                    }
                    Ok(())
                }
                "status" => {
                    ensure_daemon_running(&model_ref, max_ctx)?;
                    println!("{}", client_lora_status()?);
                    Ok(())
                }
                other => Err(format!(
                    "Unknown `lora` action '{}' (expected load, clear, status, or synth-test)",
                    other
                )),
            }
        }
        "codec-decode" => {
            if positional.len() < 2 {
                return Err("Usage: qwen-nv codec-decode <CODES_BIN> <OUT_WAV_OR_BIN> [--tts-model 1.7b]".into());
            }
            let codes_path = Path::new(&positional[0]);
            let out_path = Path::new(&positional[1]);
            let raw = fs::read(codes_path)
                .map_err(|e| format!("Failed reading {}: {}", codes_path.display(), e))?;
            // Auto-detect `qwentts.cpp` `save_raw_tensor` 12-byte header `(ndims=2, n_frames, 16)` + f32/i32 payload
            let (n_frames, codes) = if raw.len() >= 12 {
                let nd = i32::from_le_bytes(raw[0..4].try_into().unwrap());
                let d0 = i32::from_le_bytes(raw[4..8].try_into().unwrap()) as usize;
                let d1 = i32::from_le_bytes(raw[8..12].try_into().unwrap()) as usize;
                if nd == 2 && d1 == 16 && 12 + d0 * 16 * 4 == raw.len() {
                    let body = &raw[12..];
                    let mut v = Vec::with_capacity(d0 * 16);
                    for c in body.chunks_exact(4) {
                        let bits = u32::from_le_bytes(c.try_into().unwrap());
                        let as_f = f32::from_bits(bits);
                        // qwentts.cpp saves codes-full.bin as f32 values in 0.0..2048.0
                        if bits >= 0x3800_0000 && as_f.is_finite() && as_f >= 0.0 && as_f < 4096.0 {
                            v.push(as_f.round() as i32);
                        } else {
                            v.push(bits as i32);
                        }
                    }
                    (d0, v)
                } else if raw.len() % 64 == 0 {
                    let nf = raw.len() / 64;
                    let mut v = Vec::with_capacity(nf * 16);
                    for c in raw.chunks_exact(4) {
                        v.push(i32::from_le_bytes(c.try_into().unwrap()));
                    }
                    (nf, v)
                } else {
                    return Err(format!(
                        "Codes file {} size ({} B) is neither [T,16] raw i32 nor qwentts.cpp [2,T,16] tensor",
                        codes_path.display(),
                        raw.len()
                    ));
                }
            } else {
                return Err("Codes file too small".into());
            };
            ensure_daemon_running(&model_ref, max_ctx)?;
            let (samples, stats_json) =
                client_codec_decode_via_daemon(&tts_model_ref, &codes, n_frames)?;
            write_audio_output(out_path, &samples)?;
            if json_stats {
                println!("{}", stats_json);
            } else {
                eprintln!(
                    "[qwen-nv codec-decode] Wrote {} samples ({:.2}s @ 24kHz) to {} | {}",
                    samples.len(),
                    samples.len() as f64 / (TTS_SAMPLE_RATE as f64),
                    out_path.display(),
                    stats_json
                );
            }
            Ok(())
        }
        "speak" => {
            let text = positional.join(" ");
            if text.trim().is_empty() {
                return Err("Missing text to speak (usage: qwen-nv speak \"Hello world\")".into());
            }

            let tts_temp = if greedy {
                0.0
            } else if temp_explicit {
                temp
            } else {
                0.9
            };
            let params = TtsSynthParams {
                text,
                speaker: voice,
                language: lang,
                instruct,
                max_frames,
                temperature: tts_temp,
                top_k,
                top_p: 1.0,
                rep_penalty: rep_pen,
                sub_temperature: if greedy { 0.0 } else { 0.9 },
                sub_top_k: top_k,
                sub_top_p: 1.0,
                seed,
                streaming_chunks: stream_chunks,
            };

            if direct {
                install_signal_handlers();
                let mut engine = QwenMegakernelEngine::load(&model_ref, max_ctx)?;
                let mut player = if !no_play {
                    Some(CoreAudioPlayer::new(TTS_SAMPLE_RATE)?)
                } else {
                    None
                };
                let (samples, stats) = engine.synthesize_speech(&tts_model_ref, &params, None, |chunk| {
                    if let Some(p) = player.as_mut() {
                        let _ = p.push_samples(chunk);
                    }
                    true
                })?;
                if let Some(out) = &output_path {
                    write_audio_output(Path::new(out), &samples)?;
                }
                if let Some(mut p) = player {
                    p.drain();
                }
                if json_stats {
                    println!("{}", stats.to_json());
                } else {
                    eprintln!(
                        "[qwen-nv speak] {:.2}s audio ({} frames) in {:.1} ms | TTFA {:.1} ms | RTF {:.4} ({:.2}x real-time)",
                        stats.audio_sec,
                        stats.frames,
                        stats.total_ms,
                        stats.ttfa_ms,
                        stats.rtf,
                        stats.speed_x_realtime
                    );
                }
                engine.gpu.shutdown_clean();
                Ok(())
            } else {
                ensure_daemon_running(&model_ref, max_ctx)?;
                let mut player = if !no_play {
                    Some(CoreAudioPlayer::new(TTS_SAMPLE_RATE)?)
                } else {
                    None
                };
                let t_ipc = Instant::now();
                let mut client_ttfa_ms = 0.0f64;
                let mut got_first_chunk = false;

                let (samples, stats_json) = client_speak_via_daemon(
                    &tts_model_ref,
                    &params,
                    None,
                    |chunk| {
                        if !got_first_chunk {
                            got_first_chunk = true;
                            client_ttfa_ms = t_ipc.elapsed().as_secs_f64() * 1000.0;
                        }
                        if let Some(p) = player.as_mut() {
                            let _ = p.push_samples(chunk);
                        }
                        true
                    },
                )?;
                let synth_wall_ms = t_ipc.elapsed().as_secs_f64() * 1000.0;

                if let Some(out) = &output_path {
                    write_audio_output(Path::new(out), &samples)?;
                }
                if let Some(mut p) = player {
                    p.drain();
                }

                if json_stats {
                    if let Some(stripped) = stats_json.strip_suffix('}') {
                        println!(
                            "{},\"client_ttfa_ms\":{:.3},\"client_wall_ms\":{:.3}}}",
                            stripped, client_ttfa_ms, synth_wall_ms
                        );
                    } else {
                        println!("{}", stats_json);
                    }
                } else {
                    eprintln!(
                        "[qwen-nv speak] client_TTFA={:.1} ms | synth_wall={:.1} ms | samples={} ({:.2}s) | {}",
                        client_ttfa_ms,
                        synth_wall_ms,
                        samples.len(),
                        samples.len() as f64 / (TTS_SAMPLE_RATE as f64),
                        stats_json
                    );
                }
                Ok(())
            }
        }
        "tokenize" => {
            let text = positional.join(" ");
            let path = resolve_model_path(&model_ref)?;
            let gguf = GgufFile::open(&path)?;
            let tok = BpeTokenizer::from_gguf(&gguf)?;
            let cfg = QwenModelConfig::from_gguf(&gguf, &tok, max_ctx, "qwen")?;
            let ids = if raw_mode {
                tok.encode(&text)
            } else {
                let mut v = Vec::new();
                if let Some(bos) = cfg.bos_id {
                    v.push(bos);
                }
                v.extend(tok.role_tokens("user"));
                v.extend(tok.encode(&text));
                v.extend(tok.end_turn_tokens(cfg.im_end_id));
                v.extend(tok.role_tokens("assistant"));
                v
            };
            let strs: Vec<String> = ids.iter().map(|x| x.to_string()).collect();
            println!("[{}]", strs.join(", "));
            Ok(())
        }
        "run" => {
            let prompt = positional.join(" ");
            if prompt.is_empty() {
                Err("Missing prompt text (pass prompt argument)".into())
            } else if direct {
                install_signal_handlers();
                let mut engine = QwenMegakernelEngine::load(&model_ref, max_ctx)?;
                let prompt_ids = if raw_mode {
                    let mut v = Vec::new();
                    if let Some(bos) = engine.cfg.bos_id {
                        v.push(bos);
                    }
                    v.extend(engine.tok.encode(&prompt));
                    v
                } else {
                    engine.encode_chat(&[("user".into(), prompt)], false)
                };

                let tok_ptr = &engine.tok as *const BpeTokenizer;
                let mut dec = Utf8StreamDecoder::new();
                let mut out_stdout = io::stdout();

                let (out_ids, stats) =
                    engine.generate(&prompt_ids, max_tokens, temp, spec, None, |tid| {
                        if !quiet {
                            let s = unsafe { dec.push((*tok_ptr).token_bytes(tid)) };
                            if !s.is_empty() {
                                let _ = out_stdout.write_all(s.as_bytes());
                                let _ = out_stdout.flush();
                            }
                        }
                        true
                    })?;
                if !quiet {
                    let tail = dec.flush();
                    if !tail.is_empty() {
                        print!("{}", tail);
                    }
                    println!();
                }
                if json_stats {
                    println!("{}", format_stats_json(&stats, &out_ids));
                } else {
                    eprintln!(
                        "[qwen-nv] {} tokens in {:.1} ms ({:.2} tok/s decode, TTFT {:.2} ms, {} GPU launches, +{} spec tokens)",
                        stats.completion_tokens,
                        stats.decode_ms,
                        stats.decode_tok_per_s,
                        stats.ttft_ms,
                        stats.kernel_launches,
                        stats.spec_accepted_extra
                    );
                }
                engine.gpu.shutdown_clean();
                Ok(())
            } else {
                ensure_daemon_running(&model_ref, max_ctx)?;
                let t_ipc = Instant::now();
                let mut first_tok_ipc_ms = 0.0f64;
                let mut got_first = false;
                let mut out_stdout = io::stdout();
                let mode = if raw_mode { "raw" } else { "chat" };

                let stats_json = client_generate_via_daemon(
                    &prompt,
                    mode,
                    max_tokens,
                    temp,
                    spec,
                    reset_cache,
                    |_, piece| {
                        if !got_first {
                            got_first = true;
                            first_tok_ipc_ms = t_ipc.elapsed().as_secs_f64() * 1000.0;
                        }
                        if !quiet && !piece.is_empty() {
                            let _ = out_stdout.write_all(piece.as_bytes());
                            let _ = out_stdout.flush();
                        }
                    },
                )?;
                if !quiet {
                    println!();
                }
                if json_stats {
                    let wall_ms = t_ipc.elapsed().as_secs_f64() * 1000.0;
                    if let Some(stripped) = stats_json.strip_suffix('}') {
                        println!(
                            "{},\"client_ttft_ms\":{:.3},\"client_wall_ms\":{:.3}}}",
                            stripped, first_tok_ipc_ms, wall_ms
                        );
                    } else {
                        println!("{}", stats_json);
                    }
                } else {
                    eprintln!(
                        "[qwen-nv client] client_TTFT={:.2} ms | total_wall={:.2} ms | {}",
                        first_tok_ipc_ms,
                        t_ipc.elapsed().as_secs_f64() * 1000.0,
                        stats_json
                    );
                }
                Ok(())
            }
        }
        other => Err(format!("Unknown subcommand '{}'", other)),
    }
}
