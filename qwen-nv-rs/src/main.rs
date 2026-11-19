mod daemon;
mod engine;
mod gguf;
mod nv_defs;
mod nv_hw;
mod tinygpu_client;
mod tui;
mod unicode_cats;

use std::io::{self, Write};
use std::time::Instant;

use daemon::{
    client_generate_via_daemon, daemon_ping, ensure_daemon_running, format_stats_json,
    install_signal_handlers, run_daemon_foreground, run_http_server, stop_daemon,
};
use engine::QwenMegakernelEngine;
use gguf::{resolve_model_path, BpeTokenizer, GgufFile, QwenModelConfig, Utf8StreamDecoder};

fn print_usage() {
    eprintln!(
        "qwen-nv v0.2.0 — Zero-Docker, Zero-Middleware Native ARM64 Rust + GA102 (sm_86) Megakernel Engine\n\n\
         USAGE:\n  \
           qwen-nv tui  [OPTIONS]              Launch interactive TrueColor Rust Terminal UI (alias: chat)\n  \
           qwen-nv run  [OPTIONS] <PROMPT>     Generate response (uses persistent <1ms daemon by default)\n  \
           qwen-nv daemon [OPTIONS]            Run persistent GPU daemon (~/.cache/qwen-nv-rs/daemon.sock)\n  \
           qwen-nv status                      Query running GPU daemon status & warm ping latency\n  \
           qwen-nv stop                        Cleanly unload GSP-RM and stop running GPU daemon\n  \
           qwen-nv serve [OPTIONS]             Serve OpenAI-compatible HTTP/SSE API (/v1/chat/completions)\n  \
           qwen-nv tokenize [OPTIONS] <TEXT>   Offline GGUF BPE tokenizer check (zero GPU required)\n\n\
         OPTIONS:\n  \
           --model <REF>        Model alias (qwen3:4b, qwen2.5:3b, ...) or GGUF path [default: qwen3:4b]\n  \
           --max-tokens <N>     Maximum tokens to generate [default: 256]\n  \
           --temp <F>           Sampling temperature (0.0 = greedy) [default: 0.0]\n  \
           --ctx <N>            Maximum KV context length [default: 4096]\n  \
           --spec / --no-spec   Enable/disable lossless BM=4 N-gram speculative verify [default: --spec]\n  \
           --direct             Run standalone in-process instead of via persistent daemon\n  \
           --background         Spawn daemon in background and return once ready\n  \
           --reset-cache        Clear KV prefix cache before running prompt\n  \
           --raw                Encode raw text instead of ChatML (<|im_start|>user...)\n  \
           --quiet              Suppress streamed stdout text (useful for pure timing benchmarks)\n  \
           --json-stats         Print single-line JSON stats summary to stderr/stdout\n  \
           --port <PORT>        HTTP server port for `serve` [default: 8000]"
    );
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
    let mut max_tokens = 256usize;
    let mut max_tokens_explicit = false;
    let mut temp = 0.0f32;
    let mut max_ctx = 4096usize;
    let mut spec = true;
    let mut direct = false;
    let mut background = false;
    let mut reset_cache = false;
    let mut raw_mode = false;
    let mut quiet = false;
    let mut json_stats = false;
    let mut self_test = false;
    let mut port = 8000u16;
    let mut positional = Vec::<String>::new();

    let start_idx = if matches!(subcmd, "run" | "tui" | "chat" | "daemon" | "status" | "stop" | "serve" | "tokenize") {
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
            "--spec" => spec = true,
            "--no-spec" => spec = false,
            "--direct" => direct = true,
            "--background" => background = true,
            "--reset-cache" => reset_cache = true,
            "--raw" => raw_mode = true,
            "--quiet" | "-q" => quiet = true,
            "--json-stats" => json_stats = true,
            "--self-test" => self_test = true,
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
                    println!(
                        "ONLINE | model_ref={} | model_name='{}' | active_SMs={} | ping={:.1} us",
                        mref, mname, sms, us
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
            tui::run_tui(&model_ref, max_ctx, spec, temp, tui_max, self_test)
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

                let (out_ids, stats) = engine.generate(&prompt_ids, max_tokens, temp, spec, None, |tid| {
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
                    // Inject `client_wall_ms` and `client_ttft_ms` into JSON output
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
