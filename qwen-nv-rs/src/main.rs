mod asr;
mod audio;
mod daemon;
mod drafter;
mod engine;
mod gguf;
mod inkling;
mod lora;
mod mech_patch;
mod nv_defs;
mod nv_hw;
mod rag;
mod tinygpu_client;
mod topic_lora;
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
    client_asr_warm, client_codec_decode_via_daemon, client_drafter_clear, client_drafter_load,
    client_drafter_status, client_embed_via_daemon, client_eval_logits, client_extract_hooks,
    client_extract_mlp_kv, client_generate_chat_sys_via_daemon, client_generate_via_daemon,
    client_lora_clear, client_lora_load, client_lora_status, client_rag_clear, client_rag_index,
    client_rag_load, client_rag_search, client_rag_status, client_speak_via_daemon,
    client_topic_clear, client_topic_list, client_topic_load, client_topic_route,
    client_topic_status, client_transcribe_via_daemon, client_tts_warm, daemon_ping,
    ensure_daemon_running, format_stats_json, install_signal_handlers, run_daemon_foreground,
    run_http_server, stop_daemon,
};
use drafter::CalibratedNgramPrior;
use engine::QwenMegakernelEngine;
use gguf::{resolve_model_path, BpeTokenizer, GgufFile, QwenModelConfig, Utf8StreamDecoder};
use inkling::InklingEngine;
use lora::{extract_svd_peft_adapter, write_synthetic_peft_adapter};
use mech_patch::{
    apply_surgical_weight_patch, build_ablation_lora_from_bf16, kl_divergence_from_logits,
    localize_causal_directions, rollback_surgical_weight_patch,
};
use rag::{format_rag_augmented_prompt, load_jsonl_eval_queries};
use topic_lora::{
    json_find_num, solve_alphaedit_lora_layer, write_or_merge_topic_peft_lora, QmkvFile,
};
use tts::TtsSynthParams;

fn parse_i32_slice_in_line(ln: &str) -> Option<Vec<i32>> {
    let sub = if let Some(idx) = ln.find("\"tokens\"") {
        &ln[idx + 8..]
    } else {
        ln
    };
    let open = sub.find('[')?;
    let close = sub[open + 1..].find(']')?;
    let inner = &sub[open + 1..open + 1 + close];
    let mut v = Vec::new();
    for p in inner.split(',') {
        let pt = p.trim();
        if !pt.is_empty() {
            if let Ok(val) = pt.parse::<i32>() {
                v.push(val);
            }
        }
    }
    if v.is_empty() { None } else { Some(v) }
}

fn parse_i32_tokens_jsonl(path: &Path) -> Result<Vec<Vec<i32>>, String> {
    let txt = fs::read_to_string(path)
        .map_err(|e| format!("Failed reading {}: {}", path.display(), e))?;
    let mut seqs = Vec::new();
    for raw_ln in txt.lines() {
        let ln = raw_ln.trim();
        if ln.is_empty() {
            continue;
        }
        if let Some(v) = parse_i32_slice_in_line(ln) {
            seqs.push(v);
        }
    }
    if seqs.is_empty() {
        return Err(format!("No token arrays found in {}", path.display()));
    }
    Ok(seqs)
}

fn parse_kv_specs_jsonl(path: &Path) -> Result<Vec<(Vec<i32>, usize, usize)>, String> {
    let txt = fs::read_to_string(path)
        .map_err(|e| format!("Failed reading {}: {}", path.display(), e))?;
    let mut specs = Vec::new();
    for raw_ln in txt.lines() {
        let ln = raw_ln.trim();
        if ln.is_empty() {
            continue;
        }
        let Some(toks) = parse_i32_slice_in_line(ln) else {
            continue;
        };
        let pos_start = json_find_num(ln, "pos_start")
            .map(|x| x.max(0.0) as usize)
            .unwrap_or_else(|| toks.len().saturating_sub(1));
        let pos_count = json_find_num(ln, "pos_count")
            .map(|x| x.max(1.0) as usize)
            .unwrap_or(1);
        specs.push((toks, pos_start, pos_count));
    }
    if specs.is_empty() {
        return Err(format!("No token specs found in {}", path.display()));
    }
    Ok(specs)
}

fn print_usage() {
    eprintln!(
        "qwen-nv v0.7.0 — Zero-Docker, Zero-Crate Native ARM64 Rust + GA102 (sm_86) 8B Shared-Ancestor Thinker + Dynamic Spec/Persistent Switch + Mechanistic Surgery + RAG + STT + TTS Engine\n\n\
         USAGE:\n  \
           qwen-nv tui        [OPTIONS]                Launch interactive TrueColor Rust TUI (`Ctrl+R` mic, `/rag wiki`, `/talk`, `/speak`, `/lora`)\n  \
           qwen-nv run        [OPTIONS] <PROMPT>       Generate text response (uses persistent <1ms daemon by default)\n  \
           qwen-nv inkling    [OPTIONS] <PROMPT>       100%-on-RTX-3080 Inkling-2-Mini (26.4B Looped Sparse MoE, 48 logical layers)\n  \
           qwen-nv mech       <extract-hooks|extract-kv|localize|make-lora|solve-topic-lora|patch|rollback|kl-eval> Zero-retraining mechanistic surgery\n  \
           qwen-nv topic      <load|status|list|route|ask|clear> Sharded Topic Parametric LoRA Bank (`308 us` router + `44 us` swap)\n  \
           qwen-nv drafter    <load|clear|status|distill> Co-resident 100%-GPU Qwen3-0.6B neural speculative drafter\n  \
           qwen-nv embed      [OPTIONS] <TEXT>         100%-on-RTX-3080 same-model 4096-D / 2560-D embedding (`raw_eos`, `prompt_eol`, `echo`, `svd_lora`)\n  \
           qwen-nv rag        <index|load|query|ask|eval|status|clear> 100%-on-RTX-3080 `.qvec` BF16 cosine VectorDB & Wikipedia RAG\n  \
           qwen-nv transcribe [OPTIONS] <WAV|--mic>    100%-on-RTX-3080 Qwen3-ASR-1.7B Q8_0 speech-to-text (alias: listen, stt)\n  \
           qwen-nv speak      [OPTIONS] <TEXT>         100%-on-RTX-3080 Qwen3-TTS-1.7B Q8_0 streaming speech synthesis\n  \
           qwen-nv lora       <load|clear|status|extract-svd|synth-test> Dynamic <5ms Rank-r (r<=32) LoRA hot-swap & same-ancestor SVD extractor\n  \
           qwen-nv asr-warm   [OPTIONS]                Pre-load co-resident Qwen3-ASR-1.7B Q8_0 weights into VRAM\n  \
           qwen-nv tts-warm   [OPTIONS]                Pre-load co-resident Qwen3-TTS-1.7B + Code2Wav weights into VRAM\n  \
           qwen-nv codec-decode <CODES_BIN> <OUT_PATH> Run 100%-GPU 12Hz Code2Wav streaming vocoder on [T, 16] i32 codes\n  \
           qwen-nv daemon     [OPTIONS]                Run persistent GPU daemon (~/.cache/qwen-nv-rs/daemon.sock)\n  \
           qwen-nv status                              Query running GPU daemon status & warm ping latency\n  \
           qwen-nv stop                                Cleanly unload GSP-RM and stop running GPU daemon\n  \
           qwen-nv serve      [OPTIONS]                Serve OpenAI HTTP API (/v1/chat/completions, /v1/embeddings, /v1/audio/*)\n  \
           qwen-nv tokenize   [OPTIONS] <TEXT>         Offline GGUF BPE tokenizer check (zero GPU required)\n\n\
         TEXT & DAEMON OPTIONS:\n  \
           --model <REF>        Thinker GGUF alias (inkling, qwen3-8b-patched, qwen3-thinking:8b, qwen3-embed:8b, qwen3:4b, ...) or path [default: qwen3-8b-patched]\n  \
           --effort <VAL>       Inkling-2-Mini thinking effort: 0 (instant ~120 tok/s), 0.1, 0.2, 0.7, 0.9 [default: 0.2]\n  \
           --gamma <1..7>       Inkling-2-Mini DSpark speculative draft length when --spec is active [default: 3]\n  \
           --max-tokens <N>     Maximum tokens to generate [default: 256]\n  \
           --temp <F>           Sampling temperature (0.0 = greedy for text; 0.9 default for speak)\n  \
           --ctx <N>            Maximum KV context length [default: 4096]\n  \
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
           --asr-model <REF>    ASR model alias (1.7b [Q8_0 default], 1.7b-q6, 1.7b-q4, 0.6b) or GGUF path [default: 1.7b]\n  \
           --mic                Record live 16 kHz audio from macOS microphone (auto-VAD stops on ~1s pause)\n  \
           --duration <SEC>     Fixed microphone recording duration in seconds (optional; overrides auto-VAD)\n  \
           --ask                Full round-trip voice assistant: Transcribe -> [RAG] -> Qwen3-8B-Patched -> Qwen3-TTS-1.7B!\n\n\
         TTS (`speak`) OPTIONS:\n  \
           --tts-model <REF>    TTS model alias (1.7b [Q8_0 default], 1.7b-q4, 0.6b) or GGUF path [default: 1.7b]\n  \
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
    let mut model_ref = "qwen3-8b-patched".to_string();
    let mut model_explicit = false;
    let mut tts_model_ref = "1.7b".to_string();
    let mut asr_model_ref = "1.7b".to_string();
    let mut max_tokens = 256usize;
    let mut max_tokens_explicit = false;
    let mut temp = 0.0f32;
    let mut temp_explicit = false;
    let mut max_ctx = 4096usize;
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
                }
            }
            "--spec-crossover" => {
                i += 1;
                if i < args.len() {
                    if let Ok(v) = args[i].parse::<usize>() {
                        std::env::set_var("QWEN_NV_SPEC_CROSSOVER", v.to_string());
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

    let ensure_client_daemon = |mref: &str, m_exp: bool, ctx: usize| -> Result<(), String> {
        if !m_exp && daemon_ping().is_ok() {
            Ok(())
        } else {
            ensure_daemon_running(mref, ctx)
        }
    };

    match effective_cmd {
        "status" => {
            let t0 = Instant::now();
            match daemon_ping() {
                Ok((mref, mname, sms)) => {
                    let us = t0.elapsed().as_secs_f64() * 1e6;
                    let lora_info = client_lora_status().unwrap_or_default();
                    let rag_info = client_rag_status().unwrap_or_default();
                    let topic_info = client_topic_status().unwrap_or_default();
                    let dr_info = client_drafter_status().unwrap_or_default();
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
            let is_ink = InklingEngine::is_inkling_ref(&model_ref);
            let tui_spec = if is_ink && !spec_explicit { false } else { spec };
            if let Some(ref eff) = inkling_effort {
                std::env::set_var("INKLING_EFFORT", eff);
            }
            tui::run_tui(
                &model_ref,
                max_ctx,
                tui_spec,
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

                // Full Round-Trip Voice Assistant (`--ask`): STT -> [RAG] -> Qwen3-4B -> Qwen3-TTS-1.7B!
                if !json_stats {
                    println!("▸ YOU (RTX 3080 STT {:.1} ms): {}", client_wall_ms, transcript);
                }
                if transcript.trim().is_empty() {
                    return Ok(());
                }

                if let Some(ref qp) = qvec_path {
                    let _ = client_rag_load(qp)?;
                }
                let rag_k = if top_k_explicit { top_k.max(1) as usize } else { 3 };
                let rag_active = client_rag_status()
                    .map(|s| s.contains("\"active\":true"))
                    .unwrap_or(false);
                let mut rag_embed_ms = 0.0f64;
                let mut rag_gpu_us = 0.0f64;
                let mut rag_titles = Vec::<String>::new();
                let llm_prompt = if rag_active {
                    let rs = client_rag_search(&transcript, rag_k)?;
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
                    print!("◆ {}: ", model_ref);
                    let _ = io::stdout().flush();
                }
                let mut reply_text = String::new();
                let t_llm = Instant::now();
                let mut llm_ttft_ms = 0.0f64;
                let mut got_llm_tok = false;
                let llm_stats = client_generate_via_daemon(
                    &llm_prompt,
                    "chat",
                    max_tokens,
                    temp,
                    spec,
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
                    let esc_tr = transcript.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n");
                    let esc_rp = reply_text.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n");
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
                        "[qwen-nv voice-loop] STT={:.1} ms | RAG_embed={:.2} ms ({:.1} µs search) | LLM_TTFT={:.1} ms | TTS_TTFA={:.1} ms | LLM={} | TTS={}",
                        client_wall_ms, rag_embed_ms, rag_gpu_us, llm_ttft_ms, ttfa_ms, llm_stats, tts_stats
                    );
                }
                Ok(())
            }
        }
        "embed" => {
            let text = positional.join(" ");
            if text.trim().is_empty() {
                return Err("Usage: qwen-nv embed [--strategy echo] [--lora <DIR>] [--doc] <TEXT>".into());
            }
            ensure_client_daemon(&model_ref, model_explicit, max_ctx)?;
            let (vec_f32, embed_ms) =
                client_embed_via_daemon(&embed_strategy, embed_is_query, &embed_lora_path, &text)?;
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
                    ensure_client_daemon(&model_ref, model_explicit, max_ctx)?;
                    println!("{}", client_rag_status()?);
                    Ok(())
                }
                "clear" | "off" | "unload" => {
                    ensure_client_daemon(&model_ref, model_explicit, max_ctx)?;
                    println!("{}", client_rag_clear()?);
                    Ok(())
                }
                "load" => {
                    let qp = positional
                        .get(1)
                        .or(qvec_path.as_ref())
                        .ok_or_else(|| "Usage: qwen-nv rag load <INDEX.qvec>".to_string())?;
                    ensure_client_daemon(&model_ref, model_explicit, max_ctx)?;
                    println!("{}", client_rag_load(qp)?);
                    Ok(())
                }
                "index" => {
                    let corpus_jsonl = positional.get(1).ok_or_else(|| {
                        "Usage: qwen-nv rag index <CORPUS.jsonl> -o <OUT.qvec> [--strategy echo] [--lora <DIR>]".to_string()
                    })?;
                    let out_qvec = output_path
                        .as_deref()
                        .or_else(|| positional.get(2).map(|s| s.as_str()))
                        .ok_or_else(|| "Pass -o <OUT.qvec> (or second positional path) for `rag index`".to_string())?;
                    ensure_client_daemon(&model_ref, model_explicit, max_ctx)?;
                    let res = client_rag_index(
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
                        return Err("Usage: qwen-nv rag query [--qvec <INDEX.qvec>] [--top-k 3] <QUESTION>".into());
                    }
                    ensure_client_daemon(&model_ref, model_explicit, max_ctx)?;
                    if let Some(ref qp) = qvec_path {
                        let _ = client_rag_load(qp)?;
                    }
                    let st = client_rag_search(&question, rag_k)?;
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
                        return Err("Usage: qwen-nv rag ask [--qvec <INDEX.qvec>] [--top-k 3] <QUESTION>".into());
                    }
                    ensure_client_daemon(&model_ref, model_explicit, max_ctx)?;
                    if let Some(ref qp) = qvec_path {
                        let _ = client_rag_load(qp)?;
                    }
                    let st = client_rag_search(&question, rag_k)?;
                    if !quiet {
                        let cited: Vec<String> = st
                            .hits
                            .iter()
                            .map(|h| format!("{} ({:.3})", h.title, h.score))
                            .collect();
                        eprintln!(
                            "[qwen-nv rag] strategy={} | embed={:.2} ms | GPU_cosine={:.1} µs | hits=[{}]",
                            st.strategy,
                            st.embed_ms,
                            st.gpu_search_us,
                            cited.join(", ")
                        );
                    }
                    let aug_prompt = format_rag_augmented_prompt(&question, &st.hits);
                    let mut out_stdout = io::stdout();
                    let stats_json = if let Some(ref sys_p) = sys_prompt_opt {
                        client_generate_chat_sys_via_daemon(
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
                        client_generate_via_daemon(
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
                        "Usage: qwen-nv rag eval <EVAL.jsonl> [--qvec <INDEX.qvec>]".to_string()
                    })?;
                    ensure_client_daemon(&model_ref, model_explicit, max_ctx)?;
                    if let Some(ref qp) = qvec_path {
                        let _ = client_rag_load(qp)?;
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
                        let st = client_rag_search(&q.question, 10)?;
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
                    search_us_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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
        "lora" => {
            let action = positional.first().map(|s| s.as_str()).unwrap_or("status");
            match action {
                "extract-svd" => {
                    let target_ref = positional.get(1).ok_or_else(|| {
                        "Usage: qwen-nv lora extract-svd <TARGET_GGUF_OR_ALIAS> <OUT_DIR> [--rank 32] [--attn-only]".to_string()
                    })?;
                    let out_dir = positional.get(2).ok_or_else(|| {
                        "Usage: qwen-nv lora extract-svd <TARGET_GGUF_OR_ALIAS> <OUT_DIR> [--rank 32] [--attn-only]".to_string()
                    })?;
                    let svd_rank = if lora_rank_explicit { lora_rank } else { 32 };
                    let base_path = resolve_model_path(&model_ref)?;
                    let targ_path = resolve_model_path(target_ref)?;
                    let gguf = GgufFile::open(&base_path)?;
                    let tok = BpeTokenizer::from_gguf(&gguf)?;
                    let cfg = QwenModelConfig::from_gguf(&gguf, &tok, max_ctx, &model_ref)?;
                    eprintln!(
                        "[qwen-nv extract-svd] Dequantizing & running Randomized Rank-{} SVD across {} layers (include_mlp={})...",
                        svd_rank,
                        cfg.num_layers,
                        !attn_only
                    );
                    let st = extract_svd_peft_adapter(
                        &base_path,
                        &targ_path,
                        Path::new(out_dir),
                        cfg.num_layers,
                        svd_rank,
                        !attn_only,
                    )?;
                    println!(
                        "{{\"status\":\"ok\",\"out_dir\":\"{}\",\"num_layers\":{},\"rank\":{},\"include_mlp\":{},\"num_tensors\":{},\"safetensors_bytes\":{},\"safetensors_mb\":{:.2},\"attn_energy_pct\":{:.2},\"mlp_energy_pct\":{:.2},\"total_energy_pct\":{:.2},\"elapsed_s\":{:.3}}}",
                        out_dir,
                        st.num_layers,
                        st.rank,
                        st.include_mlp,
                        st.num_tensors,
                        st.safetensors_bytes,
                        (st.safetensors_bytes as f64) / (1024.0 * 1024.0),
                        st.attn_energy_pct,
                        st.mlp_energy_pct,
                        st.total_energy_pct,
                        st.elapsed_s
                    );
                    Ok(())
                }
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
                    ensure_client_daemon(&model_ref, model_explicit, max_ctx)?;
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
                    ensure_client_daemon(&model_ref, model_explicit, max_ctx)?;
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
                    ensure_client_daemon(&model_ref, model_explicit, max_ctx)?;
                    println!("{}", client_lora_status()?);
                    Ok(())
                }
                other => Err(format!(
                    "Unknown `lora` action '{}' (expected load, clear, status, extract-svd, or synth-test)",
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
            let ids = if raw_mode {
                tok.encode(&text)
            } else {
                let cfg = QwenModelConfig::from_gguf(&gguf, &tok, max_ctx, "qwen")?;
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
            let _ = stop_daemon();
            install_signal_handlers();
            let mut engine = InklingEngine::new(eff_model, max_ctx)?;
            if let Some(g) = inkling_gamma {
                engine.dspark_gamma = g.clamp(1, 7);
            }
            let eff_seed = seed.unwrap_or(42);
            let eff_temp = if temp_explicit { temp } else { 0.0 };
            let eff_max_toks = if max_tokens_explicit { max_tokens } else { 512 };
            let spec = if spec_explicit { spec } else { false };

            if inkling_pareto {
                let pareto_cases: &[(&str, &str, &str, usize)] = &[
                    (
                        "concise_factual_effort0",
                        "What are the three main architectural innovations in a looped Mixture-of-Experts transformer? Reply in 3 concise bullet points.",
                        "0",
                        384,
                    ),
                    (
                        "coding_rust_is_prime_effort0",
                        "Write a concise zero-dependency Rust function `is_prime(n: u64) -> bool`.",
                        "0",
                        384,
                    ),
                    (
                        "coding_rust_bsearch_effort0",
                        "Write a concise zero-dependency generic Rust function `lower_bound<T: Ord>(slice: &[T], target: &T) -> usize` with 3 `assert_eq!` unit test examples.",
                        "0",
                        384,
                    ),
                    (
                        "structured_json_effort0",
                        "Output ONLY a valid minified JSON object with keys `gpu`, `sm_count`, `vram_gb`, `quant`, `loop_passes`, and `logical_layers` describing an NVIDIA RTX 3080 (68 SMs, 10 GB) running Inkling-2-Mini (3 passes, 48 logical layers, IQ2).",
                        "0",
                        256,
                    ),
                    (
                        "math_amc12_nut_mix",
                        "A box contains 10 pounds of a nut mix that is 50% peanuts, 20% cashews, and 30% almonds. A second nut mix containing 20% peanuts, 40% cashews, and 40% almonds is added to the box resulting in a new nut mix that is 40% peanuts. How many pounds of cashews are now in the box? Give the exact final number.",
                        "0.2",
                        640,
                    ),
                    (
                        "math_bat_and_ball",
                        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
                        "0.2",
                        256,
                    ),
                ];

                // Strategies: `(label, enable_spec, gamma, run_all_6)`
                let strategies: &[(&str, bool, usize, bool)] = &[
                    ("persistent_b1", false, 0, true),
                    ("dspark_gamma2", true, 2, false),
                    ("dspark_gamma3", true, 3, true),
                    ("dspark_gamma4", true, 4, false),
                    ("dspark_gamma7", true, 7, false),
                ];

                let mut baseline_tokens = std::collections::HashMap::<String, Vec<i32>>::new();
                let mut rows_json = Vec::new();
                let model_file = resolve_model_path(eff_model)
                    .ok()
                    .and_then(|p| p.file_name().map(|s| s.to_string_lossy().into_owned()))
                    .unwrap_or_else(|| eff_model.to_string());

                for &(strat_name, en_spec, gam, run_all) in strategies {
                    if gam > 0 {
                        engine.dspark_gamma = gam;
                    }
                    for (c_idx, &(case_name, prompt_s, eff_str, m_toks)) in pareto_cases.iter().enumerate() {
                        // Run full 6-task suite on `persistent_b1` and `dspark_gamma3`; run 4 core speed tasks on gamma=2,4,7
                        if !run_all && c_idx == 4 {
                            continue;
                        }
                        let p_ids = engine.format_chat_prompt(prompt_s, None, Some(eff_str));
                        let mut reply = String::new();
                        let st = engine.generate_with_cancel(
                            &p_ids,
                            m_toks,
                            0.0,
                            eff_seed,
                            Some(3),
                            en_spec,
                            true,
                            None,
                            false,
                            |_, piece| {
                                reply.push_str(piece);
                                true
                            },
                        )?;
                        let lossless_match = if !en_spec {
                            baseline_tokens.insert(case_name.to_string(), st.token_ids.clone());
                            true
                        } else if let Some(ref_toks) = baseline_tokens.get(case_name) {
                            ref_toks == &st.token_ids
                        } else {
                            true
                        };
                        eprintln!(
                            "[pareto] {:16} | {:14} | {:28} | prompt={:3} ({:6.1} t/s) | gen={:3} @ {:6.2} tok/s (DSpark +{:3}/{:3}) | match={}",
                            model_file,
                            strat_name,
                            case_name,
                            st.prompt_tokens,
                            st.prefill_tok_per_s,
                            st.completion_tokens,
                            st.decode_tok_per_s,
                            st.spec_accepted_extra,
                            st.spec_verify_calls,
                            lossless_match
                        );
                        let esc_rep = reply
                            .replace('\\', "\\\\")
                            .replace('"', "\\\"")
                            .replace('\n', "\\n")
                            .replace('\r', "");
                        rows_json.push(format!(
                            "{{\"model_file\":\"{}\",\"strategy\":\"{}\",\"case\":\"{}\",\"effort\":\"{}\",\"spec\":{},\"gamma\":{},\"prompt_tokens\":{},\"completion_tokens\":{},\"prefill_ms\":{:.2},\"prefill_tok_per_s\":{:.2},\"decode_ms\":{:.2},\"decode_tok_per_s\":{:.2},\"ttft_ms\":{:.2},\"spec_verify_calls\":{},\"spec_accepted_extra\":{},\"lossless_match\":{},\"vram_weights_mb\":{:.2},\"vram_kv_conv_mb\":{:.2},\"response\":\"{}\"}}",
                            model_file,
                            strat_name,
                            case_name,
                            eff_str,
                            en_spec,
                            gam,
                            st.prompt_tokens,
                            st.completion_tokens,
                            st.prefill_ms,
                            st.prefill_tok_per_s,
                            st.decode_ms,
                            st.decode_tok_per_s,
                            st.ttft_ms,
                            st.spec_verify_calls,
                            st.spec_accepted_extra,
                            lossless_match,
                            st.vram_weights_mb,
                            st.vram_kv_conv_mb,
                            esc_rep
                        ));
                    }
                }

                let summary_json = format!(
                    "{{\"model_file\":\"{}\",\"model_name\":\"{}\",\"gpu\":\"NVIDIA GeForce RTX 3080 (GA102, sm_86, 68 SMs, 10GB GDDR6X)\",\"results\":[{}]}}",
                    model_file,
                    engine.cfg.model_name,
                    rows_json.join(",")
                );
                if let Some(ref out_p) = output_path {
                    fs::write(out_p, &summary_json).map_err(|e| e.to_string())?;
                }
                if json_stats {
                    println!("{}", summary_json);
                }
                engine.gpu.shutdown_clean();
                return Ok(());
            }

            if inkling_suite {
                let bench_cases: &[(&str, &str, &str, usize, usize)] = &[
                    (
                        "concise_factual_effort0",
                        "What are the three main architectural innovations in a looped Mixture-of-Experts transformer? Reply in 3 concise bullet points.",
                        "0",
                        3,
                        384,
                    ),
                    (
                        "math_amc12_nut_mix_thinking",
                        "A box contains 10 pounds of a nut mix that is 50% peanuts, 20% cashews, and 30% almonds. A second nut mix containing 20% peanuts, 40% cashews, and 40% almonds is added to the box resulting in a new nut mix that is 40% peanuts. How many pounds of cashews are now in the box? Give the exact final number.",
                        "0.2",
                        3,
                        640,
                    ),
                    (
                        "coding_rust_fn_effort0",
                        "Write a concise zero-dependency Rust function `is_prime(n: u64) -> bool`.",
                        "0",
                        3,
                        384,
                    ),
                    (
                        "loop_ablation_1pass",
                        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
                        "0.2",
                        1,
                        256,
                    ),
                    (
                        "loop_ablation_2pass",
                        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
                        "0.2",
                        2,
                        256,
                    ),
                    (
                        "loop_ablation_3pass",
                        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
                        "0.2",
                        3,
                        256,
                    ),
                ];

                let mut rows_json = Vec::new();
                for &(name, prompt_s, eff_str, n_pass, m_toks) in bench_cases {
                    eprintln!("\n=== [Inkling Suite: {} (passes={}, effort={}, spec={})] ===", name, n_pass, eff_str, spec);
                    let p_ids = engine.format_chat_prompt(prompt_s, sys_prompt_opt.as_deref(), Some(eff_str));
                    let mut reply = String::new();
                    let mut out_stdout = io::stdout();
                    let st = engine.generate_with_cancel(
                        &p_ids,
                        m_toks,
                        0.0,
                        eff_seed,
                        Some(n_pass),
                        spec,
                        true,
                        None,
                        false,
                        |_, piece| {
                            reply.push_str(piece);
                            if !quiet {
                                let _ = out_stdout.write_all(piece.as_bytes());
                                let _ = out_stdout.flush();
                            }
                            true
                        },
                    )?;
                    if !quiet {
                        println!();
                    }
                    let esc_rep = reply
                        .replace('\\', "\\\\")
                        .replace('"', "\\\"")
                        .replace('\n', "\\n")
                        .replace('\r', "");
                    rows_json.push(format!(
                        "{{\"case\":\"{}\",\"passes\":{},\"logical_layers\":{},\"effort\":\"{}\",\"spec\":{},\"gamma\":{},\"prompt_tokens\":{},\"completion_tokens\":{},\"prefill_ms\":{:.2},\"prefill_tok_per_s\":{:.2},\"decode_ms\":{:.2},\"decode_tok_per_s\":{:.2},\"ttft_ms\":{:.2},\"spec_verify_calls\":{},\"spec_accepted_extra\":{},\"vram_weights_mb\":{:.2},\"vram_kv_conv_mb\":{:.2},\"response\":\"{}\"}}",
                        name,
                        st.loop_passes,
                        st.logical_layers_per_tok,
                        eff_str,
                        spec && engine.dspark.is_some() && n_pass == 3,
                        engine.dspark_gamma,
                        st.prompt_tokens,
                        st.completion_tokens,
                        st.prefill_ms,
                        st.prefill_tok_per_s,
                        st.decode_ms,
                        st.decode_tok_per_s,
                        st.ttft_ms,
                        st.spec_verify_calls,
                        st.spec_accepted_extra,
                        st.vram_weights_mb,
                        st.vram_kv_conv_mb,
                        esc_rep
                    ));
                }
                let summary_json = format!("{{\"model\":\"{}\",\"gpu\":\"NVIDIA GeForce RTX 3080 (GA102, sm_86, 68 SMs, 10GB GDDR6X)\",\"results\":[{}]}}", engine.cfg.model_name, rows_json.join(","));
                if let Some(ref out_p) = output_path {
                    fs::write(out_p, &summary_json).map_err(|e| e.to_string())?;
                }
                if json_stats {
                    println!("{}", summary_json);
                }
                engine.gpu.shutdown_clean();
                return Ok(());
            }

            let prompt = positional.join(" ");
            if prompt.is_empty() {
                engine.gpu.shutdown_clean();
                return Err("Usage: qwen-nv inkling [--passes 1|2|3] [--effort 0|0.2|0.9] [--spec|--no-spec] [--dspark-gamma 1..7] [--max-tokens N] <PROMPT>".into());
            }
            let prompt_ids = if raw_mode {
                engine.tok.encode(&prompt)
            } else {
                engine.format_chat_prompt(&prompt, sys_prompt_opt.as_deref(), inkling_effort.as_deref())
            };
            let mut out_stdout = io::stdout();
            let st = engine.generate_with_cancel(
                &prompt_ids,
                eff_max_toks,
                eff_temp,
                eff_seed,
                inkling_passes,
                spec,
                true,
                None,
                raw_mode,
                |_, piece| {
                    if !quiet && !piece.is_empty() {
                        let _ = out_stdout.write_all(piece.as_bytes());
                        let _ = out_stdout.flush();
                    }
                    true
                },
            )?;
            if !quiet {
                println!();
            }
            if json_stats {
                println!(
                    "{{\"model\":\"{}\",\"loop_passes\":{},\"logical_layers\":{},\"spec\":{},\"gamma\":{},\"prompt_tokens\":{},\"completion_tokens\":{},\"prefill_ms\":{:.3},\"prefill_tok_per_s\":{:.2},\"decode_ms\":{:.3},\"decode_tok_per_s\":{:.2},\"ttft_ms\":{:.3},\"spec_verify_calls\":{},\"spec_accepted_extra\":{},\"vram_weights_mb\":{:.2},\"vram_kv_conv_mb\":{:.2}}}",
                    st.model_name,
                    st.loop_passes,
                    st.logical_layers_per_tok,
                    spec && engine.dspark.is_some() && st.loop_passes == 3,
                    engine.dspark_gamma,
                    st.prompt_tokens,
                    st.completion_tokens,
                    st.prefill_ms,
                    st.prefill_tok_per_s,
                    st.decode_ms,
                    st.decode_tok_per_s,
                    st.ttft_ms,
                    st.spec_verify_calls,
                    st.spec_accepted_extra,
                    st.vram_weights_mb,
                    st.vram_kv_conv_mb
                );
            } else if !quiet {
                eprintln!(
                    "[inkling-nv] {} passes ({} logical layers) | prompt={} ({:.1} tok/s) | gen={} @ {:.2} tok/s (DSpark +{} accepted / {} steps) | TTFT={:.1} ms | VRAM={:.0}+{:.0} MiB",
                    st.loop_passes,
                    st.logical_layers_per_tok,
                    st.prompt_tokens,
                    st.prefill_tok_per_s,
                    st.completion_tokens,
                    st.decode_tok_per_s,
                    st.spec_accepted_extra,
                    st.spec_verify_calls,
                    st.ttft_ms,
                    st.vram_weights_mb,
                    st.vram_kv_conv_mb
                );
            }
            engine.gpu.shutdown_clean();
            Ok(())
        }
        "run" => {
            let prompt = positional.join(" ");
            let is_ink = InklingEngine::is_inkling_ref(&model_ref);
            let run_spec = if is_ink && !spec_explicit { false } else { spec };
            if prompt.is_empty() {
                Err("Missing prompt text (pass prompt argument)".into())
            } else if is_ink && (direct || inkling_passes.is_some() || inkling_effort.is_some()) {
                let _ = stop_daemon();
                install_signal_handlers();
                let mut engine = InklingEngine::new(&model_ref, max_ctx)?;
                if let Some(g) = inkling_gamma {
                    engine.dspark_gamma = g.clamp(1, 7);
                }
                let prompt_ids = if raw_mode {
                    engine.tok.encode(&prompt)
                } else {
                    engine.format_chat_prompt(&prompt, sys_prompt_opt.as_deref(), inkling_effort.as_deref())
                };
                let mut out_stdout = io::stdout();
                let st = engine.generate_with_cancel(
                    &prompt_ids,
                    max_tokens,
                    temp,
                    seed.unwrap_or(42),
                    inkling_passes,
                    run_spec,
                    true,
                    None,
                    raw_mode,
                    |_, piece| {
                        if !quiet && !piece.is_empty() {
                            let _ = out_stdout.write_all(piece.as_bytes());
                            let _ = out_stdout.flush();
                        }
                        true
                    },
                )?;
                if !quiet {
                    println!();
                }
                if json_stats {
                    println!(
                        "{{\"model\":\"{}\",\"loop_passes\":{},\"logical_layers\":{},\"prompt_tokens\":{},\"completion_tokens\":{},\"prefill_ms\":{:.3},\"prefill_tok_per_s\":{:.2},\"decode_ms\":{:.3},\"decode_tok_per_s\":{:.2},\"ttft_ms\":{:.3},\"vram_weights_mb\":{:.2},\"vram_kv_conv_mb\":{:.2}}}",
                        st.model_name,
                        st.loop_passes,
                        st.logical_layers_per_tok,
                        st.prompt_tokens,
                        st.completion_tokens,
                        st.prefill_ms,
                        st.prefill_tok_per_s,
                        st.decode_ms,
                        st.decode_tok_per_s,
                        st.ttft_ms,
                        st.vram_weights_mb,
                        st.vram_kv_conv_mb
                    );
                }
                engine.gpu.shutdown_clean();
                Ok(())
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
                    engine.encode_chat(&[("user".into(), prompt)], true)
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
                ensure_client_daemon(&model_ref, model_explicit, max_ctx)?;
                let t_ipc = Instant::now();
                let mut first_tok_ipc_ms = 0.0f64;
                let mut got_first = false;
                let mut out_stdout = io::stdout();
                let mode = if raw_mode { "raw" } else { "chat" };

                let stats_json = if let Some(ref sys_p) = sys_prompt_opt {
                    client_generate_chat_sys_via_daemon(
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
                    client_generate_via_daemon(
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
                        "[qwen-nv client] client_TTFT={:.2} ms | total_wall={:.2} ms | {}",
                        first_tok_ipc_ms,
                        t_ipc.elapsed().as_secs_f64() * 1000.0,
                        stats_json
                    );
                }
                Ok(())
            }
        }
        "drafter" => {
            let action = positional.first().map(|s| s.as_str()).unwrap_or("status");
            match action {
                "status" => {
                    ensure_client_daemon(&model_ref, model_explicit, max_ctx)?;
                    println!("{}", client_drafter_status()?);
                    Ok(())
                }
                "clear" | "off" | "unload" => {
                    ensure_client_daemon(&model_ref, model_explicit, max_ctx)?;
                    println!("{}", client_drafter_clear()?);
                    Ok(())
                }
                "load" | "warm" => {
                    let d_ref = positional.get(1).map(|s| s.as_str()).unwrap_or("qwen3:0.6b");
                    ensure_client_daemon(&model_ref, model_explicit, max_ctx)?;
                    println!("{}", client_drafter_load(d_ref, &drafter_prior_path)?);
                    Ok(())
                }
                "distill" => {
                    let seqs_jsonl = positional.get(1).ok_or_else(|| {
                        "Usage: qwen-nv drafter distill <SEQS.jsonl> -o <OUT_PRIOR.bin> [--min-support 2] [--min-prob 0.85]".to_string()
                    })?;
                    let out_p = output_path.as_deref().ok_or("Pass -o <OUT_PRIOR.bin>")?;
                    let seqs = parse_i32_tokens_jsonl(Path::new(seqs_jsonl))?;
                    let prior = CalibratedNgramPrior::distill_from_sequences(&seqs, min_support, min_prob);
                    prior.save_bin(Path::new(out_p))?;
                    println!(
                        "{{\"status\":\"ok\",\"num_sequences\":{},\"map5_entries\":{},\"map4_entries\":{},\"out_path\":\"{}\"}}",
                        seqs.len(),
                        prior.map5.len(),
                        prior.map4.len(),
                        out_p
                    );
                    Ok(())
                }
                other => Err(format!(
                    "Unknown `drafter` action '{}' (expected load, clear, status, or distill)",
                    other
                )),
            }
        }
        "mech" => {
            let action = positional.first().map(|s| s.as_str()).unwrap_or("help");
            let eff_rank = if lora_rank_explicit { lora_rank } else { 1 };
            let eff_alpha = if lora_alpha_explicit { lora_alpha } else { 1.0 };
            let edit_o = if mech_down_only { false } else { true };
            let edit_dn = if mech_o_only { false } else { true };

            match action {
                "extract-hooks" => {
                    let toks_jsonl = positional.get(1).ok_or_else(|| {
                        "Usage: qwen-nv mech extract-hooks <TOKS.jsonl> -o <OUT_HOOKS.bin>".to_string()
                    })?;
                    let out_bin = output_path.as_deref().ok_or("Pass -o <OUT_HOOKS.bin>")?;
                    ensure_client_daemon(&model_ref, model_explicit, max_ctx)?;
                    let seqs = parse_i32_tokens_jsonl(Path::new(toks_jsonl))?;
                    if let Some(p) = Path::new(out_bin).parent() {
                        if !p.as_os_str().is_empty() {
                            fs::create_dir_all(p).map_err(|e| e.to_string())?;
                        }
                    }
                    let mut f = fs::File::create(out_bin).map_err(|e| e.to_string())?;
                    let t0 = Instant::now();
                    let mut floats_per_seq = 0usize;
                    for seq in &seqs {
                        let h = client_extract_hooks(seq)?;
                        floats_per_seq = h.len();
                        let mut raw = Vec::with_capacity(h.len() * 4);
                        for &v in &h {
                            raw.extend_from_slice(&v.to_bits().to_le_bytes());
                        }
                        f.write_all(&raw).map_err(|e| e.to_string())?;
                    }
                    println!(
                        "{{\"status\":\"ok\",\"num_samples\":{},\"floats_per_sample\":{},\"out_path\":\"{}\",\"elapsed_s\":{:.3}}}",
                        seqs.len(),
                        floats_per_seq,
                        out_bin,
                        t0.elapsed().as_secs_f64()
                    );
                    Ok(())
                }
                "localize" => {
                    let fail_bin = positional.get(1).ok_or_else(|| {
                        "Usage: qwen-nv mech localize <FAIL_HOOKS.bin> <CLEAN_HOOKS.bin> -o <OUT_DIRS.bin> --report <REPORT.json>".to_string()
                    })?;
                    let clean_bin = positional.get(2).ok_or_else(|| {
                        "Usage: qwen-nv mech localize <FAIL_HOOKS.bin> <CLEAN_HOOKS.bin> -o <OUT_DIRS.bin> --report <REPORT.json>".to_string()
                    })?;
                    let out_dirs = output_path.as_deref().ok_or("Pass -o <OUT_DIRS.bin>")?;
                    let out_rep = mech_report_path
                        .as_deref()
                        .unwrap_or("/Users/tim/Code/optimize-gpu/data/causal_localization.json");
                    let path = resolve_model_path(&model_ref)?;
                    let gguf = GgufFile::open(&path)?;
                    let tok = BpeTokenizer::from_gguf(&gguf)?;
                    let cfg = QwenModelConfig::from_gguf(&gguf, &tok, max_ctx, &model_ref)?;
                    let ranked = localize_causal_directions(
                        Path::new(fail_bin),
                        Path::new(clean_bin),
                        cfg.num_layers,
                        cfg.dim,
                        Path::new(out_dirs),
                        Path::new(out_rep),
                    )?;
                    let top6: Vec<String> = ranked
                        .iter()
                        .take(6)
                        .map(|s| format!("{{\"layer\":{},\"score\":{:.4},\"cos_sep\":{:.4},\"cohen_d\":{:.2}}}", s.layer, s.combined_score, s.res_cos_sep, s.res_cohen_d))
                        .collect();
                    println!(
                        "{{\"status\":\"ok\",\"dirs_bin\":\"{}\",\"report_json\":\"{}\",\"top_layers\":[{}]}}",
                        out_dirs,
                        out_rep,
                        top6.join(",")
                    );
                    Ok(())
                }
                "make-lora" => {
                    if positional.len() < 4 || mech_layers.is_empty() {
                        return Err("Usage: qwen-nv mech make-lora <BF16_DIR> <DIRS.bin> <OUT_LORA_DIR> --layers <L0,L1,...> [--rank 1] [--alpha 1.0]".into());
                    }
                    build_ablation_lora_from_bf16(
                        Path::new(&positional[1]),
                        Path::new(&positional[2]),
                        &mech_layers,
                        edit_o,
                        edit_dn,
                        eff_rank,
                        eff_alpha,
                        Path::new(&positional[3]),
                    )?;
                    println!(
                        "{{\"status\":\"ok\",\"out_lora_dir\":\"{}\",\"layers\":{:?},\"rank\":{},\"alpha\":{:.3},\"edit_o\":{},\"edit_down\":{}}}",
                        positional[3], mech_layers, eff_rank, eff_alpha, edit_o, edit_dn
                    );
                    Ok(())
                }
                "patch" => {
                    if positional.len() < 5 || mech_layers.is_empty() {
                        return Err("Usage: qwen-nv mech patch <BF16_DIR> <BASE_Q8_GGUF> <DIRS.bin> <OUT_CKPT_DIR> --layers <L0,L1,...> [--rank 1] [--alpha 1.0] [--no-rescale]".into());
                    }
                    let sum = apply_surgical_weight_patch(
                        Path::new(&positional[1]),
                        Path::new(&positional[2]),
                        Path::new(&positional[3]),
                        Path::new(&positional[4]),
                        &mech_layers,
                        edit_o,
                        edit_dn,
                        eff_rank,
                        eff_alpha,
                        mech_rescale_rows,
                    )?;
                    println!(
                        "{{\"status\":\"patched\",\"out_dir\":\"{}\",\"gguf_path\":\"{}\",\"num_tensors\":{},\"subspace_dof_params\":{},\"subspace_dof_pct\":{:.6},\"mean_rel_frobenius_delta\":{:.6},\"elapsed_s\":{:.3}}}",
                        sum.out_dir,
                        sum.gguf_path,
                        sum.edited_tensors.len(),
                        sum.subspace_dof_params,
                        sum.subspace_dof_pct,
                        sum.mean_rel_frobenius_delta,
                        sum.elapsed_s
                    );
                    Ok(())
                }
                "rollback" => {
                    let ckpt_dir = positional.get(1).ok_or("Usage: qwen-nv mech rollback <OUT_CKPT_DIR>")?;
                    let n_restored = rollback_surgical_weight_patch(Path::new(ckpt_dir))?;
                    println!("{{\"status\":\"rolled_back\",\"restored_slices\":{}}}", n_restored);
                    Ok(())
                }
                "kl-eval" => {
                    let toks_jsonl = positional.get(1).ok_or("Usage: qwen-nv mech kl-eval <CLEAN_TOKS.jsonl> --lora <LORA_DIR>")?;
                    ensure_client_daemon(&model_ref, model_explicit, max_ctx)?;
                    let seqs = parse_i32_tokens_jsonl(Path::new(toks_jsonl))?;
                    let _ = client_lora_clear()?;
                    let mut base_logits = Vec::with_capacity(seqs.len());
                    for seq in &seqs {
                        base_logits.push(client_eval_logits(seq)?);
                    }
                    if !embed_lora_path.is_empty() {
                        let _ = client_lora_load(&embed_lora_path, lora_scale)?;
                    }
                    let mut kls = Vec::with_capacity(seqs.len());
                    let mut top1_same = 0usize;
                    for (idx, seq) in seqs.iter().enumerate() {
                        let q_log = client_eval_logits(seq)?;
                        let p_log = &base_logits[idx];
                        let kl = kl_divergence_from_logits(p_log, &q_log);
                        let p_top = p_log
                            .iter()
                            .enumerate()
                            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                            .map(|x| x.0)
                            .unwrap_or(0);
                        let q_top = q_log
                            .iter()
                            .enumerate()
                            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                            .map(|x| x.0)
                            .unwrap_or(0);
                        if p_top == q_top {
                            top1_same += 1;
                        }
                        kls.push(kl);
                    }
                    if !embed_lora_path.is_empty() {
                        let _ = client_lora_clear()?;
                    }
                    let mean_kl = kls.iter().sum::<f64>() / (kls.len().max(1) as f64);
                    let max_kl = kls.iter().copied().fold(0.0f64, f64::max);
                    println!(
                        "{{\"num_positions\":{},\"mean_kl_nats\":{:.6},\"max_kl_nats\":{:.6},\"top1_agreement\":{:.4}}}",
                        seqs.len(),
                        mean_kl,
                        max_kl,
                        (top1_same as f64) / (seqs.len().max(1) as f64)
                    );
                    Ok(())
                }
                "extract-kv" => {
                    let specs_jsonl = positional.get(1).ok_or_else(|| {
                        "Usage: qwen-nv mech extract-kv <SPECS.jsonl> --layers <L0,L1,...> -o <OUT.qmkv>".to_string()
                    })?;
                    let out_qmkv = output_path.as_deref().ok_or("Pass -o <OUT.qmkv>")?;
                    if mech_layers.is_empty() {
                        return Err("Pass --layers <L0,L1,...> for `mech extract-kv`".into());
                    }
                    ensure_client_daemon(&model_ref, model_explicit, max_ctx)?;
                    let specs = parse_kv_specs_jsonl(Path::new(specs_jsonl))?;
                    let t0 = Instant::now();
                    let mut all_keys = Vec::<f32>::new();
                    let mut all_res = Vec::<f32>::new();
                    let mut total_positions = 0usize;
                    let mut hidden_dim = 0usize;
                    let mut dim = 0usize;

                    for (toks, pos_start, pos_count) in &specs {
                        let (k_f32, r_f32) = client_extract_mlp_kv(toks, &mech_layers, *pos_start, *pos_count)?;
                        dim = r_f32.len() / (*pos_count).max(1);
                        hidden_dim = k_f32.len() / ((*pos_count).max(1) * mech_layers.len());
                        all_keys.extend_from_slice(&k_f32);
                        all_res.extend_from_slice(&r_f32);
                        total_positions += *pos_count;
                    }
                    let qmkv = QmkvFile {
                        layers: mech_layers.clone(),
                        hidden_dim,
                        dim,
                        num_positions: total_positions,
                        keys: all_keys,
                        final_res: all_res,
                    };
                    qmkv.write(Path::new(out_qmkv))?;
                    println!(
                        "{{\"status\":\"ok\",\"num_sequences\":{},\"num_positions\":{},\"layers\":{:?},\"hidden_dim\":{},\"dim\":{},\"out_path\":\"{}\",\"elapsed_s\":{:.3}}}",
                        specs.len(),
                        total_positions,
                        mech_layers,
                        hidden_dim,
                        dim,
                        out_qmkv,
                        t0.elapsed().as_secs_f64()
                    );
                    Ok(())
                }
                "solve-topic-lora" => {
                    if positional.len() < 4 {
                        return Err("Usage: qwen-nv mech solve-topic-lora <BARE.qmkv> <CTX.qmkv> <OUT_LORA_DIR> --layers <L0,...> [--bg-kv <BG.qmkv>] [--gamma 50.0] [--lambda 0.001] [--rank 32] [--layer-frac 1.0] [--merge-prev <DIR>]".into());
                    }
                    let t0 = Instant::now();
                    let bare = QmkvFile::read(Path::new(&positional[1]))?;
                    let ctx = QmkvFile::read(Path::new(&positional[2]))?;
                    if bare.num_positions != ctx.num_positions || bare.dim != ctx.dim || bare.hidden_dim != ctx.hidden_dim {
                        return Err(format!(
                            "BARE.qmkv ([pos={}, dim={}, h={}) mismatches CTX.qmkv ([pos={}, dim={}, h={}])",
                            bare.num_positions, bare.dim, bare.hidden_dim,
                            ctx.num_positions, ctx.dim, ctx.hidden_dim
                        ));
                    }
                    let target_layers = if mech_layers.is_empty() {
                        bare.layers.clone()
                    } else {
                        mech_layers.clone()
                    };
                    let bg_opt = match bg_kv_path.as_deref() {
                        Some(p) => Some(QmkvFile::read(Path::new(p))?),
                        None => None,
                    };
                    let r_rows: Vec<f32> = ctx
                        .final_res
                        .iter()
                        .zip(bare.final_res.iter())
                        .map(|(&c, &b)| (c - b) * topic_layer_frac)
                        .collect();
                    let max_r = if lora_rank_explicit {
                        lora_rank.clamp(1, 32)
                    } else {
                        32
                    };
                    let mut solved = Vec::with_capacity(target_layers.len());
                    for &l in &target_layers {
                        let k1_rows = bare.extract_layer_keys(l)?;
                        let k0_rows = if let Some(ref bg) = bg_opt {
                            bg.extract_layer_keys(l)?
                        } else {
                            Vec::new()
                        };
                        solved.push(solve_alphaedit_lora_layer(
                            l,
                            &k1_rows,
                            &r_rows,
                            &k0_rows,
                            bare.hidden_dim,
                            bare.dim,
                            topic_gamma,
                            topic_lambda,
                            max_r,
                        )?);
                    }
                    let sum = write_or_merge_topic_peft_lora(
                        Path::new(&positional[3]),
                        &solved,
                        topic_merge_prev.as_deref().map(Path::new),
                        bare.dim,
                        bare.hidden_dim,
                        None,
                    )?;
                    println!(
                        "{{\"status\":\"ok\",\"out_dir\":\"{}\",\"layers\":{:?},\"rank\":{},\"num_positions\":{},\"mean_fit_rel_err\":{:.6},\"mean_bg_leak_rms\":{:.6},\"mean_delta_frob\":{:.4},\"safetensors_bytes\":{},\"elapsed_s\":{:.3}}}",
                        sum.out_dir,
                        sum.layers,
                        sum.rank,
                        bare.num_positions,
                        sum.mean_fit_rel_err,
                        sum.mean_bg_leak_rms,
                        sum.mean_delta_frob,
                        sum.safetensors_bytes,
                        t0.elapsed().as_secs_f64()
                    );
                    Ok(())
                }
                other => Err(format!(
                    "Unknown `mech` action '{}' (expected extract-hooks, extract-kv, localize, make-lora, solve-topic-lora, patch, rollback, or kl-eval)",
                    other
                )),
            }
        }
        "topic" => {
            let action = positional.first().map(|s| s.as_str()).unwrap_or("status");
            let rag_k = if top_k_explicit { top_k.max(1) as usize } else { 3 };
            let min_sc = topic_min_score.unwrap_or(-1.0);
            match action {
                "load" => {
                    let bank = positional
                        .get(1)
                        .or(topic_bank_dir.as_ref())
                        .ok_or_else(|| "Usage: qwen-nv topic load <BANK_DIR>".to_string())?;
                    ensure_client_daemon(&model_ref, model_explicit, max_ctx)?;
                    println!("{}", client_topic_load(bank)?);
                    Ok(())
                }
                "status" => {
                    ensure_client_daemon(&model_ref, model_explicit, max_ctx)?;
                    println!("{}", client_topic_status()?);
                    Ok(())
                }
                "list" => {
                    ensure_client_daemon(&model_ref, model_explicit, max_ctx)?;
                    println!("{}", client_topic_list()?);
                    Ok(())
                }
                "clear" | "off" | "unload" => {
                    ensure_client_daemon(&model_ref, model_explicit, max_ctx)?;
                    println!("{}", client_topic_clear()?);
                    Ok(())
                }
                "route" => {
                    let question = if positional.len() > 1 {
                        positional[1..].join(" ")
                    } else {
                        String::new()
                    };
                    if question.trim().is_empty() {
                        return Err("Usage: qwen-nv topic route [--bank <DIR>] [--top-k 3] [--min-score 0.30] [--keep-lora] <QUERY>".into());
                    }
                    ensure_client_daemon(&model_ref, model_explicit, max_ctx)?;
                    if let Some(ref bank) = topic_bank_dir {
                        let _ = client_topic_load(bank)?;
                    }
                    let rt = client_topic_route(&question, rag_k, min_sc, topic_keep_lora)?;
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
                        return Err("Usage: qwen-nv topic ask [--bank <DIR>] [--min-score 0.30] [--keep-lora] [--sys <TEXT>] <QUESTION>".into());
                    }
                    ensure_client_daemon(&model_ref, model_explicit, max_ctx)?;
                    if let Some(ref bank) = topic_bank_dir {
                        let _ = client_topic_load(bank)?;
                    }
                    let t_tot = Instant::now();
                    let rt = client_topic_route(&question, rag_k, min_sc, true)?;
                    let route_wall_ms = t_tot.elapsed().as_secs_f64() * 1000.0;
                    let eff_sys = sys_prompt_opt.as_deref().unwrap_or_else(|| {
                        if rt.system_prompt.is_empty() {
                            "Answer the question concisely in one short phrase or sentence."
                        } else {
                            rt.system_prompt.as_str()
                        }
                    });
                    let mut out_stdout = io::stdout();
                    let stats_json = client_generate_chat_sys_via_daemon(
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
                        let _ = client_lora_clear();
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
