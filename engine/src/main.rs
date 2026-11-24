mod cli_ops;
mod gguf;
mod hw;
mod lora;
mod mech_patch;
mod models;
mod runtime;
mod safetensors;
mod tokenizer;
mod topic_lora;
mod tui;
mod voice;

use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::thread;
use std::time::{Duration, Instant};

use cli_ops::{
    run_direct_prompt, run_drafter_subcmd, run_inkling_subcmd, run_lora_subcmd, run_mech_subcmd,
};
use gguf::{resolve_model_path, GgufFile, ModelConfig};
use models::inkling::InklingEngine;
use models::rag::{format_rag_augmented_prompt, load_jsonl_eval_queries};
use runtime::client::{DaemonControl, DaemonQuery, DaemonRagClient, DaemonStream};
use runtime::daemon::run_daemon_foreground;
use runtime::http::run_http_server;
use runtime::session::RuntimeSession;
use voice::audio::{
    read_wav_mono_16k, write_wav_pcm16, CoreAudioPlayer, CoreAudioRecorder, ASR_SAMPLE_RATE,
    TTS_SAMPLE_RATE,
};
use voice::tts::TtsSynthParams;

fn print_usage() {
    eprintln!(
        "rtx v0.8.0 — Zero-Docker, Zero-Crate Native ARM64 Rust + GA102 (sm_86) Unified Thinker (Inkling-2-Mini / Qwen3) + Co-Resident STT + TTS + Mechanistic Surgery + RAG Engine\n\n\
         SHORTCUTS:\n  \
           ./tui   [OPTIONS]                           Launch interactive TrueColor TUI (`Ctrl+R` mic, `/talk`, `/speak`, `/rag wiki`, `/lora`)\n  \
           ./speak [OPTIONS] <TEXT>                    100%-on-RTX-3080 Qwen3-TTS + 12Hz Code2Wav streaming speech synthesis\n  \
           ./listen [OPTIONS] <WAV|--mic>              100%-on-RTX-3080 Qwen3-ASR speech-to-text (add `--ask` for full voice loop)\n\n\
         USAGE:\n  \
           rtx tui        [OPTIONS]                    Launch interactive TrueColor Rust TUI (`Ctrl+R` mic, `/rag wiki`, `/talk`, `/speak`, `/lora`)\n  \
           rtx run        [OPTIONS] <PROMPT>           Generate text response (uses persistent <1ms daemon by default)\n  \
           rtx inkling    [OPTIONS] <PROMPT>           100%-on-RTX-3080 Inkling-2-Mini (26.4B Looped Sparse MoE, 48 logical layers)\n  \
           rtx mech       <extract-hooks|extract-kv|localize|make-lora|solve-topic-lora|patch|rollback|kl-eval> Zero-retraining mechanistic surgery\n  \
           rtx topic      <load|status|list|route|ask|clear> Sharded Topic Parametric LoRA Bank (`308 us` router + `44 us` swap)\n  \
           rtx drafter    <load|clear|status|distill>  Co-resident 100%-GPU Qwen3-0.6B neural speculative drafter\n  \
           rtx embed      [OPTIONS] <TEXT>             100%-on-RTX-3080 same-model 4096-D / 2560-D embedding (`raw_eos`, `prompt_eol`, `echo`, `svd_lora`)\n  \
           rtx rag        <index|load|query|ask|eval|status|clear> 100%-on-RTX-3080 `.qvec` BF16 cosine VectorDB & Wikipedia RAG\n  \
           rtx transcribe [OPTIONS] <WAV|--mic>        100%-on-RTX-3080 Qwen3-ASR Q8_0 speech-to-text (alias: listen, stt)\n  \
           rtx speak      [OPTIONS] <TEXT>             100%-on-RTX-3080 Qwen3-TTS Q8_0 + 12Hz Code2Wav streaming speech synthesis\n  \
           rtx lora       <load|clear|status|extract-svd|synth-test> Dynamic <5ms Rank-r (r<=32) LoRA hot-swap & same-ancestor SVD extractor\n  \
           rtx asr-warm   [OPTIONS]                    Pre-load co-resident Qwen3-ASR weights into VRAM\n  \
           rtx tts-warm   [OPTIONS]                    Pre-load co-resident Qwen3-TTS + Code2Wav weights into VRAM\n  \
           rtx codec-decode <CODES_BIN> <OUT_PATH>     Run 100%-GPU 12Hz Code2Wav streaming vocoder on [T, 16] i32 codes\n  \
           rtx daemon     [OPTIONS]                    Run persistent GPU daemon (~/.cache/rtx/daemon.sock)\n  \
           rtx status                                  Query running GPU daemon status & warm ping latency\n  \
           rtx stop                                    Cleanly unload GSP-RM and stop running GPU daemon\n  \
           rtx serve      [OPTIONS]                    Serve OpenAI HTTP API (/v1/chat/completions, /v1/embeddings, /v1/audio/*)\n  \
           rtx tokenize   [OPTIONS] <TEXT>             Offline GGUF BPE tokenizer check (zero GPU required)\n\n\
         TEXT & DAEMON OPTIONS:\n  \
           --model <REF>        Thinker GGUF alias (inkling, qwen3-8b-patched, qwen3-thinking:8b, qwen3-embed:8b, qwen3:4b, ...) or path [default: active daemon or RTX_MODEL / qwen3-8b-patched]\n  \
           --effort <VAL>       Inkling-2-Mini thinking effort: 0 (instant ~120 tok/s), 0.1, 0.2, 0.7, 0.9 [default: 0.2]\n  \
           --dspark-gamma <1..7> Inkling-2-Mini DSpark speculative draft length when --spec is active [default: 7]\n  \
           --max-tokens <N>     Maximum tokens to generate [default: 256]\n  \
           --temp <F>           Sampling temperature (0.0 = greedy for text; 0.9 default for speak)\n  \
           --ctx <N>            Maximum KV context length [default: 4096 (qwen) / 2048 (inkling)]\n  \
           --spec / --no-spec   Enable/disable lossless speculative verify [default: --no-spec (inkling) / --spec (qwen3)]\n  \
           --spec-crossover <N> Auto-switch BM=4 -> persistent T=1 megakernel at pos >= N (0 = never) [default: 3072 (8B) / 1536 (4B)]\n  \
           --direct             Run standalone in-process instead of via persistent daemon\n  \
           --background         Spawn daemon in background and return once ready\n  \
           --reset-cache        Clear KV prefix cache before running prompt\n  \
           --raw                Encode raw text instead of ChatML (<|im_start|>user...)\n  \
           --quiet              Suppress streamed stdout text\n  \
           --json-stats         Print single-line JSON stats summary\n  \
           --port <PORT>        HTTP server port for `serve` [default: 8000]\n\n\
         EMBEDDING & RAG (`embed` / `rag` / `lora extract-svd`) OPTIONS:\n  \
           --strategy <NAME>    Embedding strategy: raw_eos, prompt_eol, echo, svd_lora, oracle [default: echo]\n  \
           --lora <PATH>        LoRA adapter dir/safetensors for `svd_lora` strategy\n  \
           --qvec <PATH>        `.qvec` binary vector index to load for `rag query|ask|eval`\n  \
           --doc                Encode input as document passage instead of search query (`embed`)\n  \
           --attn-only          Extract SVD LoRA over attention Q/K/V/O projections only (`lora extract-svd`)\n  \
           --top-k <K>          Top-K retrieved passages for `rag query|ask` [default: 3]\n\n\
         STT (`transcribe` / `listen`) OPTIONS:\n  \
           --asr-model <REF>    ASR model alias (1.7b [Q8_0 default on Qwen], 1.7b-q6, 1.7b-q4, 0.6b [default on Inkling]) or GGUF path\n  \
           --mic                Record live 16 kHz audio from macOS microphone (auto-VAD stops on ~1s pause)\n  \
           --duration <SEC>     Fixed microphone recording duration in seconds (optional; overrides auto-VAD)\n  \
           --ask                Full round-trip voice assistant: Transcribe -> [RAG] -> Thinker (Qwen / Inkling) -> Qwen3-TTS!\n\n\
         TTS (`speak`) OPTIONS:\n  \
           --tts-model <REF>    TTS model alias (1.7b [Q8_0 default on Qwen], 1.7b-q4, 0.6b [default on Inkling]) or GGUF path\n  \
           --voice <NAME>       Speaker preset: ryan, serena, vivian, aiden, uncle_fu, ono_anna, sohee, eric, dylan [default: ryan]\n  \
           --lang <LANG>        Language tag: english, auto, chinese, japanese, korean, french, german, spanish [default: english]\n  \
           --instruct <STR>     Optional voice style instruction (CustomVoice 1.7B)\n  \
           -o, --output <PATH>  Save synthesized 24 kHz audio (.wav/.bin) or `.qvec` output (`rag index`)\n  \
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
        fs::write(path, raw)
            .map_err(|e| format!("Failed writing raw f32 {}: {}", path.display(), e))
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
    let mut model_ref = std::env::var("RTX_MODEL")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "qwen3-8b-patched".to_string());
    let mut model_explicit = false;
    let mut tts_model_ref = "1.7b".to_string();
    let mut asr_model_ref = "1.7b".to_string();
    let mut max_tokens = 256usize;
    let mut max_tokens_explicit = false;
    let mut temp = 0.0f32;
    let mut temp_explicit = false;
    let mut max_ctx = 4096usize;
    let mut max_ctx_explicit = false;
    let mut spec = true;
    let mut spec_explicit = false;
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
    let mut top_k_explicit = false;
    let mut rep_pen = 1.05f32;
    let mut max_frames = 1024usize;
    let mut stream_chunks = true;

    // LoRA & Embedding/RAG options
    let mut lora_scale = 1.0f32;
    let mut lora_rank = 16usize;
    let mut lora_rank_explicit = false;
    let mut lora_alpha = 32.0f32;
    let mut lora_alpha_explicit = false;
    let mut lora_strength = 0.25f32;
    let mut attn_only = false;
    let mut embed_strategy = "echo".to_string();
    let mut embed_lora_path = String::new();
    let mut qvec_path: Option<String> = None;
    let mut embed_is_query = true;

    // Mechanistic patching, Topic LoRA, & Drafter options
    let mut mech_layers: Vec<usize> = Vec::new();
    let mut mech_report_path: Option<String> = None;
    let mut mech_o_only = false;
    let mut mech_down_only = false;
    let mut mech_rescale_rows = true;
    let mut bg_kv_path: Option<String> = None;
    let mut topic_gamma = 50.0f32;
    let mut topic_lambda = 1e-3f32;
    let mut topic_layer_frac = 1.0f32;
    let mut topic_merge_prev: Option<String> = None;
    let mut topic_bank_dir: Option<String> = None;
    let mut topic_min_score: Option<f32> = None;
    let mut topic_keep_lora = false;
    let mut sys_prompt_opt: Option<String> = None;
    let mut drafter_prior_path = String::new();
    let mut min_support = 2u32;
    let mut min_prob = 0.85f32;
    let mut inkling_passes: Option<usize> = None;
    let mut inkling_effort: Option<String> = None;
    let mut inkling_gamma: Option<usize> = None;
    let mut inkling_suite = false;
    let mut inkling_pareto = false;

    let mut positional = Vec::<String>::new();

    let start_idx = if matches!(
        subcmd,
        "run"
            | "inkling"
            | "tui"
            | "chat"
            | "mech"
            | "topic"
            | "drafter"
            | "embed"
            | "rag"
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
                    model_explicit = true;
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
                    max_ctx_explicit = true;
                }
            }
            "--spec-crossover" => {
                i += 1;
                if i < args.len() {
                    if let Ok(v) = args[i].parse::<usize>() {
                        std::env::set_var("RTX_SPEC_CROSSOVER", v.to_string());
                    }
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
                    top_k_explicit = true;
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
                    lora_rank_explicit = true;
                }
            }
            "--alpha" => {
                i += 1;
                if i < args.len() {
                    lora_alpha = args[i].parse().unwrap_or(1.0);
                    lora_alpha_explicit = true;
                }
            }
            "--layers" => {
                i += 1;
                if i < args.len() {
                    mech_layers.clear();
                    for p in args[i].split(',') {
                        if let Ok(l) = p.trim().parse::<usize>() {
                            mech_layers.push(l);
                        }
                    }
                }
            }
            "--report" => {
                i += 1;
                if i < args.len() {
                    mech_report_path = Some(args[i].clone());
                }
            }
            "--prior" => {
                i += 1;
                if i < args.len() {
                    drafter_prior_path = args[i].clone();
                }
            }
            "--min-support" => {
                i += 1;
                if i < args.len() {
                    min_support = args[i].parse().unwrap_or(2);
                }
            }
            "--min-prob" => {
                i += 1;
                if i < args.len() {
                    min_prob = args[i].parse().unwrap_or(0.85);
                }
            }
            "--o-only" => mech_o_only = true,
            "--down-only" => mech_down_only = true,
            "--no-rescale" => mech_rescale_rows = false,
            "--strength" => {
                i += 1;
                if i < args.len() {
                    lora_strength = args[i].parse().unwrap_or(0.25);
                }
            }
            "--strategy" => {
                i += 1;
                if i < args.len() {
                    embed_strategy = args[i].clone();
                }
            }
            "--lora" => {
                i += 1;
                if i < args.len() {
                    embed_lora_path = args[i].clone();
                }
            }
            "--qvec" | "--index" => {
                i += 1;
                if i < args.len() {
                    qvec_path = Some(args[i].clone());
                }
            }
            "--bg-kv" => {
                i += 1;
                if i < args.len() {
                    bg_kv_path = Some(args[i].clone());
                }
            }
            "--gamma" => {
                i += 1;
                if i < args.len() {
                    topic_gamma = args[i].parse().unwrap_or(50.0);
                }
            }
            "--lambda" => {
                i += 1;
                if i < args.len() {
                    topic_lambda = args[i].parse().unwrap_or(1e-3);
                }
            }
            "--layer-frac" => {
                i += 1;
                if i < args.len() {
                    topic_layer_frac = args[i].parse().unwrap_or(1.0);
                }
            }
            "--merge-prev" => {
                i += 1;
                if i < args.len() {
                    topic_merge_prev = Some(args[i].clone());
                }
            }
            "--bank" => {
                i += 1;
                if i < args.len() {
                    topic_bank_dir = Some(args[i].clone());
                }
            }
            "--min-score" => {
                i += 1;
                if i < args.len() {
                    topic_min_score = args[i].parse::<f32>().ok();
                }
            }
            "--keep-lora" => topic_keep_lora = true,
            "--sys" | "--system" => {
                i += 1;
                if i < args.len() {
                    sys_prompt_opt = Some(args[i].clone());
                }
            }
            "--passes" => {
                i += 1;
                if i < args.len() {
                    inkling_passes = args[i].parse().ok();
                }
            }
            "--effort" => {
                i += 1;
                if i < args.len() {
                    inkling_effort = Some(args[i].clone());
                }
            }
            "--dspark-gamma" => {
                i += 1;
                if i < args.len() {
                    inkling_gamma = args[i].parse().ok();
                }
            }
            "--suite" => inkling_suite = true,
            "--pareto" => inkling_pareto = true,
            "--doc" | "--passage" => embed_is_query = false,
            "--query" => embed_is_query = true,
            "--attn-only" => attn_only = true,
            "--no-play" => no_play = true,
            "--play" => no_play = false,
            "--greedy" => greedy = true,
            "--no-stream-chunks" => stream_chunks = false,
            "--spec" => {
                spec = true;
                spec_explicit = true;
            }
            "--no-spec" => {
                spec = false;
                spec_explicit = true;
            }
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

    let is_ink_model = InklingEngine::is_inkling_ref(&model_ref);
    let eff_ctx = if is_ink_model && !max_ctx_explicit {
        2048
    } else {
        max_ctx
    };
    let ensure_client_daemon = |mref: &str, m_exp: bool, ctx: usize| -> Result<(), String> {
        if !m_exp && DaemonControl::ping().is_ok() {
            Ok(())
        } else {
            DaemonControl::ensure_running(mref, ctx)
        }
    };

    match effective_cmd {
        "status" => {
            let t0 = Instant::now();
            match DaemonControl::ping() {
                Ok((mref, mname, sms)) => {
                    let us = t0.elapsed().as_secs_f64() * 1e6;
                    let lora_info = DaemonControl::query(DaemonQuery::LoraStatus).unwrap_or_default();
                    let rag_info = DaemonControl::query(DaemonQuery::RagStatus).unwrap_or_default();
                    let topic_info = DaemonControl::query(DaemonQuery::TopicStatus).unwrap_or_default();
                    let dr_info = DaemonControl::query(DaemonQuery::DrafterStatus).unwrap_or_default();
                    println!(
                        "ONLINE | model_ref={} | model_name='{}' | active_SMs={} | ping={:.1} us | lora={} | rag={} | topic={} | drafter={}",
                        mref, mname, sms, us, lora_info, rag_info, topic_info, dr_info
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
            let stopped = DaemonControl::stop()?;
            if stopped {
                eprintln!("[rtx] Daemon stopped and GSP-RM unloaded cleanly.");
            } else {
                eprintln!("[rtx] No running daemon found.");
            }
            Ok(())
        }
        "daemon" => {
            if background {
                DaemonControl::ensure_running(&model_ref, eff_ctx)
            } else {
                run_daemon_foreground(&model_ref, eff_ctx)
            }
        }
        "serve" => {
            let serve_spec = if is_ink_model && !spec_explicit { false } else { spec };
            run_http_server(&model_ref, eff_ctx, port, serve_spec)
        }
        "tui" | "chat" => {
            if !model_explicit {
                if let Ok((loaded_ref, _, _)) = DaemonControl::ping() {
                    model_ref = loaded_ref;
                }
            }
            let tui_ink = InklingEngine::is_inkling_ref(&model_ref);
            let tui_ctx = if tui_ink && !max_ctx_explicit { 2048 } else { max_ctx };
            let tui_max = if max_tokens_explicit { max_tokens } else { 1024 };
            let tui_spec = if tui_ink && !spec_explicit { false } else { spec };
            if let Some(ref eff) = inkling_effort {
                std::env::set_var("RTX_EFFORT", eff);
            }
            if let Some(g) = inkling_gamma {
                std::env::set_var("RTX_GAMMA", g.clamp(1, 7).to_string());
            }
            tui::run_tui(
                &model_ref,
                tui_ctx,
                tui_spec,
                temp,
                tui_max,
                self_test,
                voice_chat_start,
                voice_self_test,
            )
        }
        "asr-warm" => {
            ensure_client_daemon(&model_ref, model_explicit, eff_ctx)?;
            println!("{}", DaemonControl::query(DaemonQuery::AsrWarm(asr_model_ref))?);
            Ok(())
        }
        "tts-warm" => {
            ensure_client_daemon(&model_ref, model_explicit, eff_ctx)?;
            println!("{}", DaemonControl::query(DaemonQuery::TtsWarm(tts_model_ref))?);
            Ok(())
        }
        "transcribe" | "listen" | "stt" => {
            let asr_lang = if lang_explicit { lang.as_str() } else { "auto" };

            if !direct {
                ensure_client_daemon(&model_ref, model_explicit, eff_ctx)?;
                if use_mic {
                    let _ = DaemonControl::query(DaemonQuery::AsrWarm(asr_model_ref.clone()))?;
                    if ask_after_stt && !no_play {
                        let _ = DaemonControl::query(DaemonQuery::TtsWarm(tts_model_ref.clone()))?;
                    }
                }
            }

            let pcm_16k = if use_mic {
                let mut rec = CoreAudioRecorder::start(ASR_SAMPLE_RATE)?;
                if let Some(sec) = mic_duration {
                    eprintln!("[rtx listen] 🎙 Recording microphone for {:.1}s (@ 16 kHz)...", sec);
                    thread::sleep(Duration::from_secs_f64(sec.max(0.2)));
                } else {
                    eprintln!("[rtx listen] 🎙 Speak your question now (auto-stops ~1s after you pause)...");
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
                    "Usage: rtx transcribe <AUDIO.wav> [--asr-model 1.7b] (or pass --mic)".to_string()
                })?;
                read_wav_mono_16k(Path::new(wav_arg))?
            };

            if direct {
                let _ = DaemonControl::stop();
                DaemonControl::install_signal_handlers();
                let mut session = RuntimeSession::new(&model_ref, eff_ctx)?;
                let lang_opt = if asr_lang.eq_ignore_ascii_case("auto") {
                    None
                } else {
                    Some(asr_lang)
                };
                let res = session.transcribe_audio(&asr_model_ref, &pcm_16k, lang_opt);
                let stats = match res {
                    Ok(s) => s,
                    Err(e) => {
                        session.shutdown_clean();
                        return Err(e);
                    }
                };
                if json_stats {
                    println!("{}", stats.to_json());
                } else {
                    println!("{}", stats.text);
                    if !quiet {
                        eprintln!(
                            "[rtx transcribe] {:.2}s audio -> {:.1} ms ({:.1}x real-time, RTF {:.4}, lang={})",
                            stats.audio_sec,
                            stats.total_ms,
                            stats.speed_x_realtime,
                            stats.rtf,
                            stats.detected_language
                        );
                    }
                }
                session.shutdown_clean();
                Ok(())
            } else {
                let t_ipc = Instant::now();
                let (transcript, stats_json) =
                    DaemonStream::transcribe(&asr_model_ref, asr_lang, &pcm_16k)?;
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
                            "[rtx transcribe] client_wall={:.1} ms | {}",
                            client_wall_ms, stats_json
                        );
                    }
                    return Ok(());
                }

                if !json_stats {
                    println!("▸ YOU (RTX 3080 STT {:.1} ms): {}", client_wall_ms, transcript);
                }
                if transcript.trim().is_empty() {
                    return Ok(());
                }

                if let Some(ref qp) = qvec_path {
                    let _ = DaemonControl::query(DaemonQuery::RagLoad(qp.clone()))?;
                }
                let rag_k = if top_k_explicit { top_k.max(1) as usize } else { 3 };
                let rag_active = DaemonControl::query(DaemonQuery::RagStatus)
                    .map(|s| s.contains("\"active\":true"))
                    .unwrap_or(false);
                let mut rag_embed_ms = 0.0f64;
                let mut rag_gpu_us = 0.0f64;
                let mut rag_titles = Vec::<String>::new();
                let llm_prompt = if rag_active {
                    let rs = DaemonRagClient::rag_search(&transcript, rag_k)?;
                    rag_embed_ms = rs.embed_ms;
                    rag_gpu_us = rs.gpu_search_us;
                    rag_titles = rs.hits.iter().map(|h| h.title.clone()).collect();
                    if !json_stats {
                        let cited: Vec<String> = rs
                            .hits
                            .iter()
                            .map(|h| format!("{} ({:.3})", h.title, h.score))
                            .collect();
                        println!(
                            "📚 RTX 3080 RAG ({}, embed {:.2} ms, GPU cosine {:.1} µs): {}",
                            rs.strategy,
                            rs.embed_ms,
                            rs.gpu_search_us,
                            cited.join(" · ")
                        );
                    }
                    format!("{} /no_think", format_rag_augmented_prompt(&transcript, &rs.hits))
                } else {
                    format!("{} /no_think", transcript)
                };

                if !json_stats {
                    let (_, loaded_name, _) = DaemonControl::ping().unwrap_or_else(|_| {
                        (model_ref.clone(), model_ref.clone(), 68)
                    });
                    print!("◆ {}: ", loaded_name);
                    let _ = io::stdout().flush();
                }
                let ask_spec = if is_ink_model && !spec_explicit { false } else { spec };
                let mut reply_text = String::new();
                let t_llm = Instant::now();
                let mut llm_ttft_ms = 0.0f64;
                let mut got_llm_tok = false;
                let llm_stats = DaemonStream::generate(
                    &llm_prompt,
                    "chat",
                    max_tokens,
                    temp,
                    ask_spec,
                    reset_cache,
                    |_, piece| {
                        if !got_llm_tok && !piece.is_empty() {
                            got_llm_tok = true;
                            llm_ttft_ms = t_llm.elapsed().as_secs_f64() * 1000.0;
                        }
                        reply_text.push_str(piece);
                        if !json_stats {
                            print!("{}", piece);
                            let _ = io::stdout().flush();
                        }
                    },
                )?;
                if !json_stats {
                    println!();
                }

                let mut player = if !no_play {
                    Some(CoreAudioPlayer::new(TTS_SAMPLE_RATE)?)
                } else {
                    None
                };
                let spoken_reply = reply_text
                    .rsplit_once("</think>")
                    .map(|(_, after)| after.trim().to_string())
                    .unwrap_or_else(|| reply_text.clone());
                let tts_params = TtsSynthParams {
                    text: spoken_reply,
                    speaker: voice,
                    language: "english".into(),
                    streaming_chunks: true,
                    ..Default::default()
                };
                let t_tts = Instant::now();
                let mut ttfa_ms = 0.0f64;
                let mut got_first = false;
                let (samples, tts_stats) =
                    DaemonStream::speak(&tts_model_ref, &tts_params, None, |chunk| {
                        if !got_first && !chunk.is_empty() {
                            got_first = true;
                            ttfa_ms = t_tts.elapsed().as_secs_f64() * 1000.0;
                        }
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
                    let titles_j: Vec<String> = rag_titles
                        .iter()
                        .map(|t| format!("\"{}\"", t.replace('\\', "\\\\").replace('"', "\\\"")))
                        .collect();
                    let esc_tr = transcript
                        .replace('\\', "\\\\")
                        .replace('"', "\\\"")
                        .replace('\n', "\\n");
                    let esc_rp = reply_text
                        .replace('\\', "\\\\")
                        .replace('"', "\\\"")
                        .replace('\n', "\\n");
                    println!(
                        "{{\"transcript\":\"{}\",\"stt_wall_ms\":{:.3},\"rag_active\":{},\"rag_embed_ms\":{:.3},\"rag_gpu_search_us\":{:.2},\"rag_titles\":[{}],\"llm_ttft_ms\":{:.3},\"reply\":\"{}\",\"tts_ttfa_ms\":{:.3},\"audio_samples\":{},\"llm_stats\":{},\"tts_stats\":{}}}",
                        esc_tr,
                        client_wall_ms,
                        rag_active,
                        rag_embed_ms,
                        rag_gpu_us,
                        titles_j.join(","),
                        llm_ttft_ms,
                        esc_rp,
                        ttfa_ms,
                        samples.len(),
                        llm_stats,
                        tts_stats
                    );
                } else if !quiet {
                    eprintln!(
                        "[rtx voice-loop] STT={:.1} ms | RAG_embed={:.2} ms ({:.1} µs search) | LLM_TTFT={:.1} ms | TTS_TTFA={:.1} ms | LLM={} | TTS={}",
                        client_wall_ms, rag_embed_ms, rag_gpu_us, llm_ttft_ms, ttfa_ms, llm_stats, tts_stats
                    );
                }
                Ok(())
            }
        }
        "embed" => {
            let text = positional.join(" ");
            if text.trim().is_empty() {
                return Err("Usage: rtx embed [--strategy echo] [--lora <DIR>] [--doc] <TEXT>".into());
            }
            ensure_client_daemon(&model_ref, model_explicit, eff_ctx)?;
            let (vec_f32, embed_ms) =
                DaemonRagClient::embed(&embed_strategy, embed_is_query, &embed_lora_path, &text)?;
            if let Some(ref out) = output_path {
                let mut raw = Vec::with_capacity(vec_f32.len() * 4);
                for &v in &vec_f32 {
                    raw.extend_from_slice(&v.to_bits().to_le_bytes());
                }
                fs::write(out, raw).map_err(|e| format!("Failed writing {}: {}", out, e))?;
            }
            let norm: f64 = vec_f32.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt();
            let prev: Vec<String> = vec_f32.iter().take(8).map(|v| format!("{:.6}", v)).collect();
            println!(
                "{{\"dim\":{},\"strategy\":\"{}\",\"is_query\":{},\"embed_ms\":{:.3},\"l2_norm\":{:.6},\"preview\":[{}]}}",
                vec_f32.len(),
                embed_strategy,
                embed_is_query,
                embed_ms,
                norm,
                prev.join(",")
            );
            Ok(())
        }
        "rag" => {
            let action = positional.first().map(|s| s.as_str()).unwrap_or("status");
            let rag_k = if top_k_explicit { top_k.max(1) as usize } else { 3 };
            match action {
                "status" => {
                    ensure_client_daemon(&model_ref, model_explicit, eff_ctx)?;
                    println!("{}", DaemonControl::query(DaemonQuery::RagStatus)?);
                    Ok(())
                }
                "clear" | "off" | "unload" => {
                    ensure_client_daemon(&model_ref, model_explicit, eff_ctx)?;
                    println!("{}", DaemonControl::query(DaemonQuery::RagClear)?);
                    Ok(())
                }
                "load" => {
                    let qp = positional
                        .get(1)
                        .or(qvec_path.as_ref())
                        .ok_or_else(|| "Usage: rtx rag load <INDEX.qvec>".to_string())?;
                    ensure_client_daemon(&model_ref, model_explicit, eff_ctx)?;
                    println!("{}", DaemonControl::query(DaemonQuery::RagLoad(qp.clone()))?);
                    Ok(())
                }
                "index" => {
                    let corpus_jsonl = positional.get(1).ok_or_else(|| {
                        "Usage: rtx rag index <CORPUS.jsonl> -o <OUT.qvec> [--strategy echo] [--lora <DIR>]".to_string()
                    })?;
                    let out_qvec = output_path
                        .as_deref()
                        .or_else(|| positional.get(2).map(|s| s.as_str()))
                        .ok_or_else(|| "Pass -o <OUT.qvec> (or second positional path) for `rag index`".to_string())?;
                    ensure_client_daemon(&model_ref, model_explicit, eff_ctx)?;
                    let res = DaemonRagClient::rag_index(
                        &embed_strategy,
                        corpus_jsonl,
                        out_qvec,
                        &embed_lora_path,
                        true,
                    )?;
                    println!("{}", res);
                    Ok(())
                }
                "query" | "search" => {
                    let question = if positional.len() > 1 {
                        positional[1..].join(" ")
                    } else {
                        String::new()
                    };
                    if question.trim().is_empty() {
                        return Err("Usage: rtx rag query [--qvec <INDEX.qvec>] [--top-k 3] <QUESTION>".into());
                    }
                    ensure_client_daemon(&model_ref, model_explicit, eff_ctx)?;
                    if let Some(ref qp) = qvec_path {
                        let _ = DaemonControl::query(DaemonQuery::RagLoad(qp.clone()))?;
                    }
                    let st = DaemonRagClient::rag_search(&question, rag_k)?;
                    if json_stats {
                        let hits_j: Vec<String> = st
                            .hits
                            .iter()
                            .map(|h| {
                                let tt = h.title.replace('\\', "\\\\").replace('"', "\\\"");
                                let tx = h
                                    .text
                                    .replace('\\', "\\\\")
                                    .replace('"', "\\\"")
                                    .replace('\n', "\\n");
                                format!(
                                    "{{\"id\":{},\"score\":{:.6},\"title\":\"{}\",\"text\":\"{}\"}}",
                                    h.id, h.score, tt, tx
                                )
                            })
                            .collect();
                        println!(
                            "{{\"strategy\":\"{}\",\"num_docs\":{},\"embed_ms\":{:.3},\"gpu_search_us\":{:.2},\"total_ms\":{:.3},\"hits\":[{}]}}",
                            st.strategy,
                            st.num_docs,
                            st.embed_ms,
                            st.gpu_search_us,
                            st.total_ms,
                            hits_j.join(",")
                        );
                    } else {
                        println!(
                            "📚 RTX 3080 Vector Search ({}, {} docs | embed {:.2} ms | GPU BF16 cosine {:.1} µs):",
                            st.strategy, st.num_docs, st.embed_ms, st.gpu_search_us
                        );
                        for (rank_i, h) in st.hits.iter().enumerate() {
                            let snippet: String = h.text.chars().take(180).collect();
                            println!(
                                "  #{:<2} [id={:<4} cos={:.4}] {} — {}...",
                                rank_i + 1,
                                h.id,
                                h.score,
                                h.title,
                                snippet.replace('\n', " ")
                            );
                        }
                    }
                    Ok(())
                }
                "ask" => {
                    let question = if positional.len() > 1 {
                        positional[1..].join(" ")
                    } else {
                        String::new()
                    };
                    if question.trim().is_empty() {
                        return Err("Usage: rtx rag ask [--qvec <INDEX.qvec>] [--top-k 3] <QUESTION>".into());
                    }
                    ensure_client_daemon(&model_ref, model_explicit, eff_ctx)?;
                    if let Some(ref qp) = qvec_path {
                        let _ = DaemonControl::query(DaemonQuery::RagLoad(qp.clone()))?;
                    }
                    let st = DaemonRagClient::rag_search(&question, rag_k)?;
                    if !quiet {
                        let cited: Vec<String> = st
                            .hits
                            .iter()
                            .map(|h| format!("{} ({:.3})", h.title, h.score))
                            .collect();
                        eprintln!(
                            "[rtx rag] strategy={} | embed={:.2} ms | GPU_cosine={:.1} µs | hits=[{}]",
                            st.strategy,
                            st.embed_ms,
                            st.gpu_search_us,
                            cited.join(", ")
                        );
                    }
                    let aug_prompt = format_rag_augmented_prompt(&question, &st.hits);
                    let mut out_stdout = io::stdout();
                    let stats_json = if let Some(ref sys_p) = sys_prompt_opt {
                        DaemonStream::generate_chat_sys(
                            sys_p,
                            &aug_prompt,
                            max_tokens,
                            temp,
                            spec,
                            true,
                            |_, piece| {
                                if !quiet && !piece.is_empty() {
                                    let _ = out_stdout.write_all(piece.as_bytes());
                                    let _ = out_stdout.flush();
                                }
                            },
                        )?
                    } else {
                        DaemonStream::generate(
                            &aug_prompt,
                            "chat",
                            max_tokens,
                            temp,
                            spec,
                            true,
                            |_, piece| {
                                if !quiet && !piece.is_empty() {
                                    let _ = out_stdout.write_all(piece.as_bytes());
                                    let _ = out_stdout.flush();
                                }
                            },
                        )?
                    };
                    if !quiet {
                        println!();
                    }
                    if json_stats {
                        println!("{}", stats_json);
                    }
                    Ok(())
                }
                "eval" => {
                    let eval_jsonl = positional.get(1).ok_or_else(|| {
                        "Usage: rtx rag eval <EVAL.jsonl> [--qvec <INDEX.qvec>]".to_string()
                    })?;
                    ensure_client_daemon(&model_ref, model_explicit, eff_ctx)?;
                    if let Some(ref qp) = qvec_path {
                        let _ = DaemonControl::query(DaemonQuery::RagLoad(qp.clone()))?;
                    }
                    let queries = load_jsonl_eval_queries(Path::new(eval_jsonl))?;
                    let n_q = queries.len();
                    let t_eval = Instant::now();
                    let mut r1 = 0usize;
                    let mut r5 = 0usize;
                    let mut r10 = 0usize;
                    let mut mrr10_sum = 0.0f64;
                    let mut embed_times = Vec::with_capacity(n_q);
                    let mut search_us_times = Vec::with_capacity(n_q);
                    let mut active_strat = String::new();
                    let mut active_docs = 0usize;

                    for q in &queries {
                        let st = DaemonRagClient::rag_search(&q.question, 10)?;
                        if active_strat.is_empty() {
                            active_strat = st.strategy.clone();
                            active_docs = st.num_docs;
                        }
                        embed_times.push(st.embed_ms);
                        search_us_times.push(st.gpu_search_us);
                        if let Some(pos) = st.hits.iter().position(|h| h.id == q.target_id) {
                            if pos == 0 {
                                r1 += 1;
                            }
                            if pos < 5 {
                                r5 += 1;
                            }
                            if pos < 10 {
                                r10 += 1;
                                mrr10_sum += 1.0 / ((pos + 1) as f64);
                            }
                        }
                    }
                    embed_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    search_us_times
                        .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let mean_emb = embed_times.iter().sum::<f64>() / (n_q as f64);
                    let med_emb = embed_times[n_q / 2];
                    let mean_us = search_us_times.iter().sum::<f64>() / (n_q as f64);
                    let med_us = search_us_times[n_q / 2];
                    let wall_s = t_eval.elapsed().as_secs_f64();

                    println!(
                        "{{\"strategy\":\"{}\",\"num_docs\":{},\"num_queries\":{},\"recall_at_1\":{:.4},\"recall_at_5\":{:.4},\"recall_at_10\":{:.4},\"mrr_at_10\":{:.4},\"r1_hits\":{},\"r5_hits\":{},\"r10_hits\":{},\"mean_embed_ms\":{:.3},\"median_embed_ms\":{:.3},\"mean_gpu_search_us\":{:.2},\"median_gpu_search_us\":{:.2},\"eval_wall_s\":{:.3}}}",
                        active_strat,
                        active_docs,
                        n_q,
                        (r1 as f64) / (n_q as f64),
                        (r5 as f64) / (n_q as f64),
                        (r10 as f64) / (n_q as f64),
                        mrr10_sum / (n_q as f64),
                        r1,
                        r5,
                        r10,
                        mean_emb,
                        med_emb,
                        mean_us,
                        med_us,
                        wall_s
                    );
                    Ok(())
                }
                other => Err(format!(
                    "Unknown `rag` action '{}' (expected index, load, clear, status, query, ask, or eval)",
                    other
                )),
            }
        }
        "lora" => run_lora_subcmd(
            &positional,
            &model_ref,
            model_explicit,
            eff_ctx,
            lora_rank,
            lora_rank_explicit,
            lora_alpha,
            lora_strength,
            lora_scale,
            attn_only,
        ),
        "codec-decode" => {
            if positional.len() < 2 {
                return Err("Usage: rtx codec-decode <CODES_BIN> <OUT_WAV_OR_BIN> [--tts-model 1.7b]".into());
            }
            let codes_path = Path::new(&positional[0]);
            let out_path = Path::new(&positional[1]);
            let raw = fs::read(codes_path)
                .map_err(|e| format!("Failed reading {}: {}", codes_path.display(), e))?;
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
            ensure_client_daemon(&model_ref, model_explicit, eff_ctx)?;
            let (samples, stats_json) =
                DaemonStream::codec_decode(&tts_model_ref, &codes, n_frames)?;
            write_audio_output(out_path, &samples)?;
            if json_stats {
                println!("{}", stats_json);
            } else {
                eprintln!(
                    "[rtx codec-decode] Wrote {} samples ({:.2}s @ 24kHz) to {} | {}",
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
                return Err("Missing text to speak (usage: ./speak \"Hello world\" or rtx speak \"Hello world\")".into());
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
                let _ = DaemonControl::stop();
                DaemonControl::install_signal_handlers();
                let mut session = RuntimeSession::new(&model_ref, eff_ctx)?;
                let mut player = if !no_play {
                    Some(CoreAudioPlayer::new(TTS_SAMPLE_RATE)?)
                } else {
                    None
                };
                let res = session.synthesize_speech(&tts_model_ref, &params, None, |chunk| {
                    if let Some(p) = player.as_mut() {
                        let _ = p.push_samples(chunk);
                    }
                    true
                });
                let (samples, stats) = match res {
                    Ok(pair) => pair,
                    Err(e) => {
                        session.shutdown_clean();
                        return Err(e);
                    }
                };
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
                        "[rtx speak] {:.2}s audio ({} frames) in {:.1} ms | TTFA {:.1} ms | RTF {:.4} ({:.2}x real-time)",
                        stats.audio_sec,
                        stats.frames,
                        stats.total_ms,
                        stats.ttfa_ms,
                        stats.rtf,
                        stats.speed_x_realtime
                    );
                }
                session.shutdown_clean();
                Ok(())
            } else {
                ensure_client_daemon(&model_ref, model_explicit, eff_ctx)?;
                let mut player = if !no_play {
                    Some(CoreAudioPlayer::new(TTS_SAMPLE_RATE)?)
                } else {
                    None
                };
                let t_ipc = Instant::now();
                let mut client_ttfa_ms = 0.0f64;
                let mut got_first_chunk = false;

                let (samples, stats_json) =
                    DaemonStream::speak(&tts_model_ref, &params, None, |chunk| {
                        if !got_first_chunk {
                            got_first_chunk = true;
                            client_ttfa_ms = t_ipc.elapsed().as_secs_f64() * 1000.0;
                        }
                        if let Some(p) = player.as_mut() {
                            let _ = p.push_samples(chunk);
                        }
                        true
                    })?;
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
                        "[rtx speak] client_TTFA={:.1} ms | synth_wall={:.1} ms | samples={} ({:.2}s) | {}",
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
            let tok = gguf.build_tokenizer()?;
            let ids = if raw_mode {
                tok.encode(&text)
            } else if tok.preset() == "inkling2" {
                let eff =
                    InklingEngine::normalize_effort(inkling_effort.as_deref().unwrap_or("0.2"));
                let mut s = format!(
                    "<|message_system|><|content_text|>Thinking effort level: {}<|end_message|><|message_user|><|content_text|>{}<|end_message|>",
                    eff,
                    text.trim()
                );
                if eff == "0" {
                    s.push_str("<|message_model|><|content_text|>");
                }
                tok.encode(&s)
            } else {
                let cfg = ModelConfig::from_gguf(&gguf, &tok, max_ctx, "qwen")?;
                let mut v = Vec::new();
                if let Some(bos) = cfg.bos_id {
                    v.push(bos);
                }
                v.extend(tok.role_tokens("user"));
                v.extend(tok.encode(&text));
                v.extend(tok.end_turn_tokens(cfg.im_end_id));
                v.extend(tok.role_tokens("assistant"));
                if cfg.thinking_template {
                    v.extend(tok.encode("<think>\n"));
                }
                v
            };
            let strs: Vec<String> = ids.iter().map(|x| x.to_string()).collect();
            println!("[{}]", strs.join(", "));
            Ok(())
        }
        "inkling" => {
            let eff_model = if model_explicit {
                model_ref.as_str()
            } else {
                "inkling"
            };
            let ink_ctx = if max_ctx_explicit { max_ctx } else { 2048 };
            run_inkling_subcmd(
                &positional,
                eff_model,
                ink_ctx,
                inkling_gamma,
                seed,
                temp,
                temp_explicit,
                max_tokens,
                max_tokens_explicit,
                spec,
                spec_explicit,
                inkling_pareto,
                inkling_suite,
                inkling_passes,
                inkling_effort.as_deref(),
                sys_prompt_opt.as_deref(),
                raw_mode,
                quiet,
                json_stats,
                output_path.as_deref(),
            )
        }
        "run" => {
            let prompt = positional.join(" ");
            let run_spec = if is_ink_model && !spec_explicit { false } else { spec };
            if prompt.is_empty() {
                Err("Missing prompt text (pass prompt argument)".into())
            } else if direct
                || (is_ink_model && (inkling_passes.is_some() || inkling_effort.is_some()))
            {
                run_direct_prompt(
                    &model_ref,
                    eff_ctx,
                    &prompt,
                    raw_mode,
                    sys_prompt_opt.as_deref(),
                    inkling_effort.as_deref(),
                    inkling_passes,
                    inkling_gamma,
                    max_tokens,
                    temp,
                    seed.unwrap_or(42),
                    run_spec,
                    quiet,
                    json_stats,
                )
            } else {
                ensure_client_daemon(&model_ref, model_explicit, eff_ctx)?;
                if is_ink_model {
                    if let Some(g) = inkling_gamma {
                        let _ = DaemonControl::query(DaemonQuery::InklingSetGamma(g));
                    }
                }
                let t_ipc = Instant::now();
                let mut first_tok_ipc_ms = 0.0f64;
                let mut got_first = false;
                let mut out_stdout = io::stdout();
                let mode = if raw_mode { "raw" } else { "chat" };

                let stats_json = if let Some(ref sys_p) = sys_prompt_opt {
                    DaemonStream::generate_chat_sys(
                        sys_p,
                        &prompt,
                        max_tokens,
                        temp,
                        run_spec,
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
                    )?
                } else {
                    DaemonStream::generate(
                        &prompt,
                        mode,
                        max_tokens,
                        temp,
                        run_spec,
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
                    )?
                };
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
                        "[rtx client] client_TTFT={:.2} ms | total_wall={:.2} ms | {}",
                        first_tok_ipc_ms,
                        t_ipc.elapsed().as_secs_f64() * 1000.0,
                        stats_json
                    );
                }
                Ok(())
            }
        }
        "drafter" => run_drafter_subcmd(
            &positional,
            &model_ref,
            model_explicit,
            eff_ctx,
            &drafter_prior_path,
            output_path.as_deref(),
            min_support,
            min_prob,
        ),
        "mech" => run_mech_subcmd(
            &positional,
            &model_ref,
            model_explicit,
            eff_ctx,
            output_path.as_deref(),
            mech_report_path.as_deref(),
            &mech_layers,
            mech_o_only,
            mech_down_only,
            mech_rescale_rows,
            lora_rank,
            lora_rank_explicit,
            lora_alpha,
            lora_alpha_explicit,
            &embed_lora_path,
            lora_scale,
            bg_kv_path.as_deref(),
            topic_gamma,
            topic_lambda,
            topic_layer_frac,
            topic_merge_prev.as_deref(),
        ),
        "topic" => {
            let action = positional.first().map(|s| s.as_str()).unwrap_or("status");
            let rag_k = if top_k_explicit { top_k.max(1) as usize } else { 3 };
            let min_sc = topic_min_score.unwrap_or(-1.0);
            match action {
                "load" => {
                    let bank = positional
                        .get(1)
                        .or(topic_bank_dir.as_ref())
                        .ok_or_else(|| "Usage: rtx topic load <BANK_DIR>".to_string())?;
                    ensure_client_daemon(&model_ref, model_explicit, eff_ctx)?;
                    println!("{}", DaemonControl::query(DaemonQuery::TopicLoad(bank.clone()))?);
                    Ok(())
                }
                "status" => {
                    ensure_client_daemon(&model_ref, model_explicit, eff_ctx)?;
                    println!("{}", DaemonControl::query(DaemonQuery::TopicStatus)?);
                    Ok(())
                }
                "list" => {
                    ensure_client_daemon(&model_ref, model_explicit, eff_ctx)?;
                    println!("{}", DaemonControl::query(DaemonQuery::TopicList)?);
                    Ok(())
                }
                "clear" | "off" | "unload" => {
                    ensure_client_daemon(&model_ref, model_explicit, eff_ctx)?;
                    println!("{}", DaemonControl::query(DaemonQuery::TopicClear)?);
                    Ok(())
                }
                "route" => {
                    let question = if positional.len() > 1 {
                        positional[1..].join(" ")
                    } else {
                        String::new()
                    };
                    if question.trim().is_empty() {
                        return Err("Usage: rtx topic route [--bank <DIR>] [--top-k 3] [--min-score 0.30] [--keep-lora] <QUERY>".into());
                    }
                    ensure_client_daemon(&model_ref, model_explicit, eff_ctx)?;
                    if let Some(ref bank) = topic_bank_dir {
                        let _ = DaemonControl::query(DaemonQuery::TopicLoad(bank.clone()))?;
                    }
                    let rt = DaemonRagClient::topic_route(&question, rag_k, min_sc, topic_keep_lora)?;
                    println!("{}", rt.raw_json);
                    Ok(())
                }
                "ask" => {
                    let question = if positional.len() > 1 {
                        positional[1..].join(" ")
                    } else {
                        String::new()
                    };
                    if question.trim().is_empty() {
                        return Err("Usage: rtx topic ask [--bank <DIR>] [--min-score 0.30] [--keep-lora] [--sys <TEXT>] <QUESTION>".into());
                    }
                    ensure_client_daemon(&model_ref, model_explicit, eff_ctx)?;
                    if let Some(ref bank) = topic_bank_dir {
                        let _ = DaemonControl::query(DaemonQuery::TopicLoad(bank.clone()))?;
                    }
                    let t_tot = Instant::now();
                    let rt = DaemonRagClient::topic_route(&question, rag_k, min_sc, true)?;
                    let route_wall_ms = t_tot.elapsed().as_secs_f64() * 1000.0;
                    let eff_sys = sys_prompt_opt.as_deref().unwrap_or_else(|| {
                        if rt.system_prompt.is_empty() {
                            "Answer the question concisely in one short phrase or sentence."
                        } else {
                            rt.system_prompt.as_str()
                        }
                    });
                    let mut out_stdout = io::stdout();
                    let stats_json = DaemonStream::generate_chat_sys(
                        eff_sys,
                        &question,
                        max_tokens,
                        temp,
                        spec,
                        true,
                        |_, piece| {
                            if !quiet && !piece.is_empty() {
                                let _ = out_stdout.write_all(piece.as_bytes());
                                let _ = out_stdout.flush();
                            }
                        },
                    )?;
                    if !topic_keep_lora && rt.matched {
                        let _ = DaemonControl::query(DaemonQuery::LoraClear);
                    }
                    let total_wall_ms = t_tot.elapsed().as_secs_f64() * 1000.0;
                    if !quiet {
                        println!();
                    }
                    if json_stats {
                        let esc_topic = rt.topic.replace('\\', "\\\\").replace('"', "\\\"");
                        let esc_slug = rt.slug.replace('\\', "\\\\").replace('"', "\\\"");
                        let esc_lora = rt.lora_dir.replace('\\', "\\\\").replace('"', "\\\"");
                        println!(
                            "{{\"matched\":{},\"topic\":\"{}\",\"slug\":\"{}\",\"score\":{:.6},\"lora_dir\":\"{}\",\"embed_ms\":{:.3},\"gpu_search_us\":{:.2},\"lora_swap_ms\":{:.3},\"route_wall_ms\":{:.3},\"topic_total_wall_ms\":{:.3},\"gen_stats\":{}}}",
                            rt.matched,
                            esc_topic,
                            esc_slug,
                            rt.score,
                            esc_lora,
                            rt.embed_ms,
                            rt.gpu_search_us,
                            rt.swap_ms,
                            route_wall_ms,
                            total_wall_ms,
                            stats_json
                        );
                    }
                    Ok(())
                }
                other => Err(format!(
                    "Unknown `topic` action '{}' (expected load, status, list, clear, route, or ask)",
                    other
                )),
            }
        }
        other => Err(format!("Unknown subcommand '{}'", other)),
    }
}
