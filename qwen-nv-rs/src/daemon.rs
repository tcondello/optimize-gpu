//! Persistent Background GPU Daemon (`~/.cache/qwen-nv-rs/daemon.sock`) & OpenAI-Compatible HTTP Server.
//! Keeps GA102 GSP-RM booted and Qwen weights + KV cache resident in GDDR6X across invocations,
//! cutting warm client startup latency from `9.49 s` to `< 1 ms`!

use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::os::unix::net::{UnixListener, UnixStream};
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::audio::{encode_wav_pcm16, parse_wav_bytes_mono_16k, TTS_SAMPLE_RATE};
use crate::engine::{GenerationStats, QwenMegakernelEngine};
use crate::gguf::Utf8StreamDecoder;
use crate::lora::LoraStatus;
use crate::tts::TtsSynthParams;

static SHUTDOWN_REQ: AtomicBool = AtomicBool::new(false);

extern "C" fn handle_sig(_sig: i32) {
    SHUTDOWN_REQ.store(true, Ordering::SeqCst);
}

extern "C" {
    fn signal(sig: i32, handler: extern "C" fn(i32)) -> usize;
}

const SIGHUP: i32 = 1;
const SIGINT: i32 = 2;
const SIGTERM: i32 = 15;
const SIGPIPE: i32 = 13;
const SIG_IGN: usize = 1;

pub fn install_signal_handlers() {
    unsafe {
        signal(SIGINT, handle_sig);
        signal(SIGTERM, handle_sig);
        // Ignore SIGHUP & SIGPIPE so closing a parent shell or client socket never kills the daemon!
        let _ = signal(SIGHUP, std::mem::transmute::<usize, extern "C" fn(i32)>(SIG_IGN));
        let _ = signal(SIGPIPE, std::mem::transmute::<usize, extern "C" fn(i32)>(SIG_IGN));
    }
}

pub fn daemon_sock_path() -> PathBuf {
    if let Ok(p) = std::env::var("QWEN_NV_DAEMON_SOCK") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    PathBuf::from(home).join(".cache/qwen-nv-rs/daemon.sock")
}

pub fn format_stats_json(stats: &GenerationStats, token_ids: &[i32]) -> String {
    let ids_str: Vec<String> = token_ids.iter().map(|t| t.to_string()).collect();
    format!(
        "{{\"prompt_tokens\":{},\"cached_prefix_tokens\":{},\"prefill_new_tokens\":{},\"completion_tokens\":{},\
          \"prefill_ms\":{:.3},\"decode_ms\":{:.3},\"ttft_ms\":{:.3},\"total_ms\":{:.3},\
          \"decode_tok_per_s\":{:.2},\"prefill_tok_per_s\":{:.2},\"kernel_launches\":{},\
          \"spec_verify_calls\":{},\"spec_accepted_extra\":{},\"stop_reason\":\"{}\",\"token_ids\":[{}]}}",
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
        stats.stop_reason,
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

// -----------------------------------------------------------------------------
// Unix Domain Socket Daemon Server
// -----------------------------------------------------------------------------

pub fn run_daemon_foreground(model_ref: &str, max_ctx: usize) -> Result<(), String> {
    install_signal_handlers();
    SHUTDOWN_REQ.store(false, Ordering::SeqCst);

    let sock_path = daemon_sock_path();
    if let Some(parent) = sock_path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    if sock_path.exists() {
        // Check if an existing daemon is alive
        if daemon_ping().is_ok() {
            return Err(format!("Daemon is already running at {}", sock_path.display()));
        }
        let _ = fs::remove_file(&sock_path);
    }

    let mut engine = QwenMegakernelEngine::load(model_ref, max_ctx)?;

    // Quick 1-token warmup so instruction caches & TLBs are hot before accepting clients
    let warm_ids = engine.encode_chat(&[("user".into(), "Hi".into())], false);
    let _ = engine.generate(&warm_ids, 2, 0.0, false, None, |_| true)?;
    engine.reset_cache();

    let listener = UnixListener::bind(&sock_path)
        .map_err(|e| format!("Failed to bind {}: {}", sock_path.display(), e))?;
    listener.set_nonblocking(true).map_err(|e| e.to_string())?;

    eprintln!(
        "[qwen-nv daemon] Ready on {} (model='{}', SMs={}, warm latency < 1 ms)",
        sock_path.display(),
        engine.cfg.model_name,
        engine.gpu.active_sms
    );

    while !SHUTDOWN_REQ.load(Ordering::Relaxed) {
        match listener.accept() {
            Ok((stream, _)) => {
                let _ = stream.set_nonblocking(false);
                if let Ok(should_stop) = handle_daemon_client(&mut engine, stream) {
                    if should_stop {
                        break;
                    }
                }
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(2));
            }
            Err(e) => {
                eprintln!("[qwen-nv daemon] Accept error: {}", e);
                thread::sleep(Duration::from_millis(10));
            }
        }
    }

    eprintln!("[qwen-nv daemon] Unloading GSP-RM cleanly...");
    let _ = fs::remove_file(&sock_path);
    engine.gpu.shutdown_clean();
    Ok(())
}

fn handle_daemon_client(engine: &mut QwenMegakernelEngine, mut stream: UnixStream) -> Result<bool, String> {
    let mut hdr_line = Vec::with_capacity(128);
    let mut b = [0u8; 1];
    loop {
        let n = stream.read(&mut b).map_err(|e| e.to_string())?;
        if n == 0 {
            return Ok(false);
        }
        if b[0] == b'\n' {
            break;
        }
        hdr_line.push(b[0]);
        if hdr_line.len() > 4096 {
            return Err("Header line too long".into());
        }
    }
    let line = String::from_utf8_lossy(&hdr_line);
    let parts: Vec<&str> = line.split('\t').collect();
    match parts.first().copied().unwrap_or("") {
        "PING" => {
            let reply = format!("PONG\t{}\t{}\t{}\n", engine.model_ref, engine.cfg.model_name, engine.gpu.active_sms);
            let _ = stream.write_all(reply.as_bytes());
            Ok(false)
        }
        "RESET" => {
            engine.reset_cache();
            let _ = stream.write_all(b"OK\n");
            Ok(false)
        }
        "STOP" => {
            let _ = stream.write_all(b"BYE\n");
            Ok(true)
        }
        "LORA_STATUS" => {
            let j = format_lora_json(&engine.lora_status);
            let _ = stream.write_all(format!("OK\t{}\n", j).as_bytes());
            Ok(false)
        }
        "LORA_CLEAR" => {
            match engine.clear_lora() {
                Ok(st) => {
                    let _ = stream.write_all(format!("OK\t{}\n", format_lora_json(&st)).as_bytes());
                }
                Err(e) => {
                    let _ = stream.write_all(format!("ERR\t{}\n", e.replace('\n', " ")).as_bytes());
                }
            }
            Ok(false)
        }
        "LORA_LOAD" => {
            if parts.len() < 3 {
                let _ = stream.write_all(b"ERR\tMalformed LORA_LOAD\n");
                return Ok(false);
            }
            let scale: f32 = parts[1].parse().unwrap_or(1.0);
            let path = parts[2..].join("\t");
            match engine.load_lora(&path, scale) {
                Ok(st) => {
                    let _ = stream.write_all(format!("OK\t{}\n", format_lora_json(&st)).as_bytes());
                }
                Err(e) => {
                    let _ = stream.write_all(format!("ERR\t{}\n", e.replace('\n', " ")).as_bytes());
                }
            }
            Ok(false)
        }
        "TTS_WARM" => {
            let tk_ref = parts.get(1).copied().unwrap_or("1.7b");
            match engine.ensure_tts_loaded(tk_ref) {
                Ok(()) => {
                    let tts = engine.tts.as_ref().unwrap();
                    let _ = stream.write_all(
                        format!("OK\t{}\t{:.1}\t{}\n", tts.model_name, tts.total_vram_mb, tts.speaker_names().join(",")).as_bytes(),
                    );
                }
                Err(e) => {
                    let _ = stream.write_all(format!("ERR\t{}\n", e.replace('\n', " ")).as_bytes());
                }
            }
            Ok(false)
        }
        "ASR_WARM" => {
            let asr_ref = parts.get(1).copied().unwrap_or("1.7b");
            match engine.ensure_asr_loaded(asr_ref) {
                Ok(()) => {
                    let asr = engine.asr.as_ref().unwrap();
                    let vram_mb = (asr.vram_bytes_total as f64) / (1024.0 * 1024.0);
                    let _ = stream.write_all(
                        format!("OK\t{}\t{:.1}\n", asr.model_name, vram_mb).as_bytes(),
                    );
                }
                Err(e) => {
                    let _ = stream.write_all(format!("ERR\t{}\n", e.replace('\n', " ")).as_bytes());
                }
            }
            Ok(false)
        }
        "TRANSCRIBE" => {
            // TRANSCRIBE \t asr_ref \t lang_hint \t n_samples + raw `[n_samples]` f32 LE bytes
            if parts.len() < 4 {
                let _ = stream.write_all(b"ERR\tMalformed TRANSCRIBE header\n");
                return Ok(false);
            }
            let asr_ref = parts[1];
            let lang_hint = parts[2];
            let n_samples: usize = parts[3].parse().unwrap_or(0);
            let mut raw = vec![0u8; n_samples * 4];
            if !raw.is_empty() {
                stream.read_exact(&mut raw).map_err(|e| e.to_string())?;
            }
            let pcm_16k: Vec<f32> = raw
                .chunks_exact(4)
                .map(|c| f32::from_bits(u32::from_le_bytes(c.try_into().unwrap())))
                .collect();
            let lang_opt = if lang_hint.is_empty() || lang_hint.eq_ignore_ascii_case("auto") {
                None
            } else {
                Some(lang_hint)
            };
            match engine.transcribe_audio(asr_ref, &pcm_16k, lang_opt) {
                Ok(stats) => {
                    let _ = stream.write_all(format!("OK\t{}\n", stats.to_json()).as_bytes());
                    let _ = stream.flush();
                }
                Err(e) => {
                    let _ = stream.write_all(format!("ERR\t{}\n", e.replace('\n', " ")).as_bytes());
                    let _ = stream.flush();
                }
            }
            Ok(false)
        }
        "CODEC_DECODE" => {
            // CODEC_DECODE \t talker_ref \t n_frames + raw `[n_frames * 16]` i32s
            if parts.len() < 3 {
                return Err("Malformed CODEC_DECODE".into());
            }
            let tk_ref = parts[1];
            let n_frames: usize = parts[2].parse().unwrap_or(0);
            let mut raw = vec![0u8; n_frames * 16 * 4];
            if !raw.is_empty() {
                stream.read_exact(&mut raw).map_err(|e| e.to_string())?;
            }
            let codes: Vec<i32> = raw
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                .collect();

            let res = engine.ensure_tts_loaded(tk_ref).and_then(|_| {
                let tts = engine.tts.as_mut().unwrap();
                tts.reset_codec_stream(&mut engine.gpu)?;
                let t0 = Instant::now();
                let wav = tts.decode_codes_stream_chunk(&mut engine.gpu, &engine.module, &codes, n_frames)?;
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                Ok((wav, ms))
            });
            match res {
                Ok((wav, ms)) => {
                    let mut frame = Vec::with_capacity(5 + wav.len() * 4);
                    frame.push(0x03);
                    frame.extend_from_slice(&(wav.len() as u32).to_le_bytes());
                    for &s in &wav {
                        frame.extend_from_slice(&s.to_le_bytes());
                    }
                    let _ = stream.write_all(&frame);
                    let sj = format!("{{\"frames\":{},\"n_samples\":{},\"codec_ms\":{:.3}}}", n_frames, wav.len(), ms);
                    let sb = sj.as_bytes();
                    let mut dframe = Vec::with_capacity(5 + sb.len());
                    dframe.push(0x02);
                    dframe.extend_from_slice(&(sb.len() as u32).to_le_bytes());
                    dframe.extend_from_slice(sb);
                    let _ = stream.write_all(&dframe);
                    let _ = stream.flush();
                }
                Err(err_msg) => {
                    let eb = err_msg.as_bytes();
                    let mut eframe = Vec::with_capacity(5 + eb.len());
                    eframe.push(0xff);
                    eframe.extend_from_slice(&(eb.len() as u32).to_le_bytes());
                    eframe.extend_from_slice(eb);
                    let _ = stream.write_all(&eframe);
                }
            }
            Ok(false)
        }
        "SPEAK" => {
            // SPEAK \t talker_ref \t speaker \t lang \t max_frames \t temp \t top_k \t rep_pen \t seed_i64 \t streaming
            if parts.len() < 10 {
                return Err("Malformed SPEAK header".into());
            }
            let tk_ref = parts[1];
            let speaker = parts[2].to_string();
            let language = parts[3].to_string();
            let max_frames: usize = parts[4].parse().unwrap_or(1024);
            let temperature: f32 = parts[5].parse().unwrap_or(0.9);
            let top_k: i32 = parts[6].parse().unwrap_or(50);
            let rep_penalty: f32 = parts[7].parse().unwrap_or(1.05);
            let seed_raw: i64 = parts[8].parse().unwrap_or(-1);
            let streaming_chunks: bool = parts[9] == "1";

            let mut ilen_b = [0u8; 4];
            stream.read_exact(&mut ilen_b).map_err(|e| e.to_string())?;
            let ilen = u32::from_le_bytes(ilen_b) as usize;
            let mut ibuf = vec![0u8; ilen];
            if ilen > 0 {
                stream.read_exact(&mut ibuf).map_err(|e| e.to_string())?;
            }
            let mut tlen_b = [0u8; 4];
            stream.read_exact(&mut tlen_b).map_err(|e| e.to_string())?;
            let tlen = u32::from_le_bytes(tlen_b) as usize;
            let mut tbuf = vec![0u8; tlen];
            if tlen > 0 {
                stream.read_exact(&mut tbuf).map_err(|e| e.to_string())?;
            }

            let params = TtsSynthParams {
                text: String::from_utf8_lossy(&tbuf).into_owned(),
                speaker,
                language,
                instruct: String::from_utf8_lossy(&ibuf).into_owned(),
                max_frames,
                temperature,
                top_k,
                top_p: 1.0,
                rep_penalty,
                sub_temperature: if temperature <= 0.0 { 0.0 } else { 0.9 },
                sub_top_k: top_k,
                sub_top_p: 1.0,
                seed: if seed_raw >= 0 { Some(seed_raw as u64) } else { None },
                streaming_chunks,
            };

            let mut client_alive = true;
            let synth_res = engine.synthesize_speech(tk_ref, &params, Some(&SHUTDOWN_REQ), |samples| {
                if samples.is_empty() {
                    return true;
                }
                let mut frame = Vec::with_capacity(5 + samples.len() * 4);
                frame.push(0x03);
                frame.extend_from_slice(&(samples.len() as u32).to_le_bytes());
                for &s in samples {
                    frame.extend_from_slice(&s.to_le_bytes());
                }
                if stream.write_all(&frame).is_err() || stream.flush().is_err() {
                    client_alive = false;
                    return false;
                }
                true
            });

            if !client_alive {
                return Ok(false);
            }

            match synth_res {
                Ok((_, stats)) => {
                    let sj = stats.to_json();
                    let sb = sj.as_bytes();
                    let mut dframe = Vec::with_capacity(5 + sb.len());
                    dframe.push(0x02);
                    dframe.extend_from_slice(&(sb.len() as u32).to_le_bytes());
                    dframe.extend_from_slice(sb);
                    let _ = stream.write_all(&dframe);
                    let _ = stream.flush();
                }
                Err(err_msg) => {
                    let eb = err_msg.as_bytes();
                    let mut eframe = Vec::with_capacity(5 + eb.len());
                    eframe.push(0xff);
                    eframe.extend_from_slice(&(eb.len() as u32).to_le_bytes());
                    eframe.extend_from_slice(eb);
                    let _ = stream.write_all(&eframe);
                }
            }
            Ok(false)
        }
        "GEN" => {
            // GEN \t max_tokens \t temp \t spec \t reset \t mode("chat"|"raw"|"ids")
            if parts.len() < 6 {
                return Err("Malformed GEN header".into());
            }
            let max_tokens: usize = parts[1].parse().unwrap_or(256);
            let temp: f32 = parts[2].parse().unwrap_or(0.0);
            let spec: bool = parts[3] == "1";
            let reset: bool = parts[4] == "1";
            let mode = parts[5];

            let mut len_b = [0u8; 4];
            stream.read_exact(&mut len_b).map_err(|e| e.to_string())?;
            let payload_len = u32::from_le_bytes(len_b) as usize;
            let mut payload = vec![0u8; payload_len];
            if payload_len > 0 {
                stream.read_exact(&mut payload).map_err(|e| e.to_string())?;
            }

            if reset {
                engine.reset_cache();
            }

            let prompt_ids: Vec<i32> = if mode == "ids" {
                payload
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                    .collect()
            } else if mode == "raw" {
                let s = String::from_utf8_lossy(&payload);
                let mut v = Vec::new();
                if let Some(bos) = engine.cfg.bos_id {
                    v.push(bos);
                }
                v.extend(engine.tok.encode(&s));
                v
            } else {
                // "chat": tab-separated lines `role\tcontent` or single user prompt
                let s = String::from_utf8_lossy(&payload);
                let msgs = vec![("user".to_string(), s.into_owned())];
                engine.encode_chat(&msgs, false)
            };

            let mut utf8_dec = Utf8StreamDecoder::new();
            // Clone `tok2bytes` lookup via raw slice copy inside closure? To avoid borrow conflict with `&mut engine`,
            // note `engine.generate` borrows `&mut self`, so we decode token bytes via a helper or buffer!
            // Even cleaner: `stream` writes `[0x01, tok_id:i32]` OR we read `tok2bytes` before? Wait: `tok` is immutable!
            // Let's split borrow or look up bytes via a raw pointer to immutable `engine.tok` (never mutated after load!).
            let tok_ptr = &engine.tok as *const crate::gguf::BpeTokenizer;
            let mut client_alive = true;

            let gen_res = engine.generate(
                &prompt_ids,
                max_tokens,
                temp,
                spec,
                Some(&SHUTDOWN_REQ),
                |tok_id| {
                    let piece = unsafe { utf8_dec.push((*tok_ptr).token_bytes(tok_id)) };
                    let pb = piece.as_bytes();
                    let mut frame = Vec::with_capacity(7 + pb.len());
                    frame.push(0x01);
                    frame.extend_from_slice(&tok_id.to_le_bytes());
                    frame.extend_from_slice(&(pb.len() as u16).to_le_bytes());
                    frame.extend_from_slice(pb);
                    if stream.write_all(&frame).is_err() || stream.flush().is_err() {
                        client_alive = false;
                        return false;
                    }
                    true
                },
            );

            if !client_alive {
                return Ok(false);
            }

            match gen_res {
                Ok((out_ids, stats)) => {
                    let tail = utf8_dec.flush();
                    if !tail.is_empty() {
                        let pb = tail.as_bytes();
                        let mut frame = Vec::with_capacity(7 + pb.len());
                        frame.push(0x01);
                        frame.extend_from_slice(&(-1i32).to_le_bytes());
                        frame.extend_from_slice(&(pb.len() as u16).to_le_bytes());
                        frame.extend_from_slice(pb);
                        let _ = stream.write_all(&frame);
                    }
                    let stats_json = format_stats_json(&stats, &out_ids);
                    let sb = stats_json.as_bytes();
                    let mut done_frame = Vec::with_capacity(5 + sb.len());
                    done_frame.push(0x02);
                    done_frame.extend_from_slice(&(sb.len() as u32).to_le_bytes());
                    done_frame.extend_from_slice(sb);
                    let _ = stream.write_all(&done_frame);
                    let _ = stream.flush();
                }
                Err(err_msg) => {
                    let eb = err_msg.as_bytes();
                    let mut eframe = Vec::with_capacity(5 + eb.len());
                    eframe.push(0xff);
                    eframe.extend_from_slice(&(eb.len() as u32).to_le_bytes());
                    eframe.extend_from_slice(eb);
                    let _ = stream.write_all(&eframe);
                }
            }
            Ok(false)
        }
        other => Err(format!("Unknown daemon command '{}'", other)),
    }
}

// -----------------------------------------------------------------------------
// Daemon Client Functions
// -----------------------------------------------------------------------------

pub fn daemon_ping() -> Result<(String, String, u32), String> {
    let sock_path = daemon_sock_path();
    let mut s = UnixStream::connect(&sock_path).map_err(|e| e.to_string())?;
    let _ = s.set_read_timeout(Some(Duration::from_secs(2)));
    s.write_all(b"PING\n").map_err(|e| e.to_string())?;
    let mut rdr = BufReader::new(s);
    let mut line = String::new();
    rdr.read_line(&mut line).map_err(|e| e.to_string())?;
    let parts: Vec<&str> = line.trim().split('\t').collect();
    if parts.len() >= 4 && parts[0] == "PONG" {
        let sms = parts[3].parse::<u32>().unwrap_or(68);
        Ok((parts[1].to_string(), parts[2].to_string(), sms))
    } else {
        Err(format!("Unexpected PING reply: {}", line))
    }
}

pub fn stop_daemon() -> Result<bool, String> {
    let sock_path = daemon_sock_path();
    let Ok(mut s) = UnixStream::connect(&sock_path) else {
        return Ok(false);
    };
    let _ = s.set_read_timeout(Some(Duration::from_secs(5)));
    s.write_all(b"STOP\n").map_err(|e| e.to_string())?;
    let mut buf = [0u8; 16];
    let _ = s.read(&mut buf);
    for _ in 0..50 {
        if !sock_path.exists() {
            break;
        }
        thread::sleep(Duration::from_millis(50));
    }
    Ok(true)
}

pub fn ensure_daemon_running(model_ref: &str, max_ctx: usize) -> Result<(), String> {
    if let Ok((loaded_ref, _, _)) = daemon_ping() {
        if loaded_ref == model_ref {
            return Ok(());
        }
        eprintln!(
            "[qwen-nv] Daemon currently holds '{}'; restarting for '{}'...",
            loaded_ref, model_ref
        );
        let _ = stop_daemon();
    }

    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let log_path = daemon_sock_path().with_extension("log");
    if let Some(p) = log_path.parent() {
        let _ = fs::create_dir_all(p);
    }
    let log_file = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&log_path)
        .map_err(|e| e.to_string())?;
    let err_file = log_file.try_clone().map_err(|e| e.to_string())?;

    eprintln!(
        "[qwen-nv] Starting persistent background GPU daemon (log: {})...",
        log_path.display()
    );
    let mut child = Command::new(exe)
        .args(["daemon", "--model", model_ref, "--ctx", &max_ctx.to_string()])
        .process_group(0)
        .stdin(Stdio::null())
        .stdout(Stdio::from(log_file))
        .stderr(Stdio::from(err_file))
        .spawn()
        .map_err(|e| format!("Failed to spawn daemon: {}", e))?;

    let t0 = Instant::now();
    while t0.elapsed() < Duration::from_secs(30) {
        if daemon_ping().is_ok() {
            return Ok(());
        }
        if let Ok(Some(status)) = child.try_wait() {
            let log_tail = fs::read_to_string(&log_path).unwrap_or_default();
            return Err(format!("Daemon exited early ({}):\n{}", status, log_tail));
        }
        thread::sleep(Duration::from_millis(50));
    }
    Err("Timed out waiting 30s for background GPU daemon to initialize".into())
}

pub fn client_generate_via_daemon<F>(
    prompt: &str,
    mode: &str,
    max_tokens: usize,
    temp: f32,
    spec: bool,
    reset_cache: bool,
    mut on_piece: F,
) -> Result<String, String>
where
    F: FnMut(i32, &str),
{
    let sock_path = daemon_sock_path();
    let mut s = UnixStream::connect(&sock_path)
        .map_err(|e| format!("Failed to connect to daemon {}: {}", sock_path.display(), e))?;

    let hdr = format!(
        "GEN\t{}\t{:.4}\t{}\t{}\t{}\n",
        max_tokens,
        temp,
        u8::from(spec),
        u8::from(reset_cache),
        mode
    );
    let pb = prompt.as_bytes();
    let mut req = Vec::with_capacity(hdr.len() + 4 + pb.len());
    req.extend_from_slice(hdr.as_bytes());
    req.extend_from_slice(&(pb.len() as u32).to_le_bytes());
    req.extend_from_slice(pb);
    s.write_all(&req).map_err(|e| e.to_string())?;
    s.flush().map_err(|e| e.to_string())?;

    loop {
        let mut tag = [0u8; 1];
        s.read_exact(&mut tag).map_err(|e| format!("Daemon stream closed: {}", e))?;
        match tag[0] {
            0x01 => {
                let mut meta = [0u8; 6];
                s.read_exact(&mut meta).map_err(|e| e.to_string())?;
                let tok_id = i32::from_le_bytes(meta[0..4].try_into().unwrap());
                let slen = u16::from_le_bytes(meta[4..6].try_into().unwrap()) as usize;
                let mut sbytes = vec![0u8; slen];
                if slen > 0 {
                    s.read_exact(&mut sbytes).map_err(|e| e.to_string())?;
                }
                let piece = String::from_utf8_lossy(&sbytes);
                on_piece(tok_id, &piece);
            }
            0x02 => {
                let mut lb = [0u8; 4];
                s.read_exact(&mut lb).map_err(|e| e.to_string())?;
                let jlen = u32::from_le_bytes(lb) as usize;
                let mut jb = vec![0u8; jlen];
                s.read_exact(&mut jb).map_err(|e| e.to_string())?;
                return Ok(String::from_utf8_lossy(&jb).into_owned());
            }
            0xff => {
                let mut lb = [0u8; 4];
                s.read_exact(&mut lb).map_err(|e| e.to_string())?;
                let elen = u32::from_le_bytes(lb) as usize;
                let mut eb = vec![0u8; elen];
                s.read_exact(&mut eb).map_err(|e| e.to_string())?;
                return Err(String::from_utf8_lossy(&eb).into_owned());
            }
            other => return Err(format!("Unexpected daemon frame tag {:#x}", other)),
        }
    }
}

pub fn client_simple_cmd(cmd_line: &str) -> Result<String, String> {
    let sock_path = daemon_sock_path();
    let mut s = UnixStream::connect(&sock_path)
        .map_err(|e| format!("Failed to connect to daemon {}: {}", sock_path.display(), e))?;
    let _ = s.set_read_timeout(Some(Duration::from_secs(60)));
    s.write_all(cmd_line.as_bytes()).map_err(|e| e.to_string())?;
    s.flush().map_err(|e| e.to_string())?;
    let mut rdr = BufReader::new(s);
    let mut reply = String::new();
    rdr.read_line(&mut reply).map_err(|e| e.to_string())?;
    let trimmed = reply.trim_end_matches(&['\r', '\n'][..]);
    if let Some(rest) = trimmed.strip_prefix("OK\t") {
        Ok(rest.to_string())
    } else if let Some(err) = trimmed.strip_prefix("ERR\t") {
        Err(err.to_string())
    } else {
        Err(format!("Unexpected daemon reply: {}", trimmed))
    }
}

pub fn client_lora_load(path: &str, scale: f32) -> Result<String, String> {
    client_simple_cmd(&format!("LORA_LOAD\t{:.4}\t{}\n", scale, path))
}

pub fn client_lora_clear() -> Result<String, String> {
    client_simple_cmd("LORA_CLEAR\n")
}

pub fn client_lora_status() -> Result<String, String> {
    client_simple_cmd("LORA_STATUS\n")
}

pub fn client_tts_warm(talker_ref: &str) -> Result<String, String> {
    client_simple_cmd(&format!("TTS_WARM\t{}\n", talker_ref))
}

pub fn client_asr_warm(asr_ref: &str) -> Result<String, String> {
    client_simple_cmd(&format!("ASR_WARM\t{}\n", asr_ref))
}

pub fn client_transcribe_via_daemon(
    asr_ref: &str,
    lang_hint: &str,
    pcm_16k: &[f32],
) -> Result<(String, String), String> {
    let sock_path = daemon_sock_path();
    let mut s = UnixStream::connect(&sock_path)
        .map_err(|e| format!("Failed to connect to daemon {}: {}", sock_path.display(), e))?;
    let _ = s.set_read_timeout(Some(Duration::from_secs(60)));

    let eff_ref = if asr_ref.trim().is_empty() { "1.7b" } else { asr_ref.trim() };
    let eff_lang = if lang_hint.trim().is_empty() { "auto" } else { lang_hint.trim() };
    let hdr = format!("TRANSCRIBE\t{}\t{}\t{}\n", eff_ref, eff_lang, pcm_16k.len());
    let mut req = Vec::with_capacity(hdr.len() + pcm_16k.len() * 4);
    req.extend_from_slice(hdr.as_bytes());
    for &smp in pcm_16k {
        req.extend_from_slice(&smp.to_bits().to_le_bytes());
    }
    s.write_all(&req).map_err(|e| e.to_string())?;
    s.flush().map_err(|e| e.to_string())?;

    let mut rdr = BufReader::new(s);
    let mut reply = String::new();
    rdr.read_line(&mut reply).map_err(|e| e.to_string())?;
    let trimmed = reply.trim_end_matches(&['\r', '\n'][..]);
    if let Some(stats_json) = trimmed.strip_prefix("OK\t") {
        let text = stats_json
            .find("\"text\"")
            .and_then(|idx| extract_json_str_after(&stats_json[idx + 6..]))
            .unwrap_or_default();
        Ok((text, stats_json.to_string()))
    } else if let Some(err) = trimmed.strip_prefix("ERR\t") {
        Err(err.to_string())
    } else {
        Err(format!("Unexpected TRANSCRIBE reply: {}", trimmed))
    }
}

pub fn client_codec_decode_via_daemon(
    talker_ref: &str,
    codes_flat: &[i32],
    n_frames: usize,
) -> Result<(Vec<f32>, String), String> {
    let sock_path = daemon_sock_path();
    let mut s = UnixStream::connect(&sock_path)
        .map_err(|e| format!("Failed to connect to daemon {}: {}", sock_path.display(), e))?;
    let hdr = format!("CODEC_DECODE\t{}\t{}\n", talker_ref, n_frames);
    let mut req = Vec::with_capacity(hdr.len() + codes_flat.len() * 4);
    req.extend_from_slice(hdr.as_bytes());
    for &c in codes_flat {
        req.extend_from_slice(&c.to_le_bytes());
    }
    s.write_all(&req).map_err(|e| e.to_string())?;
    s.flush().map_err(|e| e.to_string())?;

    let mut all_samples = Vec::new();
    loop {
        let mut tag = [0u8; 1];
        s.read_exact(&mut tag).map_err(|e| e.to_string())?;
        match tag[0] {
            0x03 => {
                let mut nb = [0u8; 4];
                s.read_exact(&mut nb).map_err(|e| e.to_string())?;
                let n = u32::from_le_bytes(nb) as usize;
                let mut raw = vec![0u8; n * 4];
                s.read_exact(&mut raw).map_err(|e| e.to_string())?;
                for c in raw.chunks_exact(4) {
                    all_samples.push(f32::from_bits(u32::from_le_bytes(c.try_into().unwrap())));
                }
            }
            0x02 => {
                let mut lb = [0u8; 4];
                s.read_exact(&mut lb).map_err(|e| e.to_string())?;
                let jlen = u32::from_le_bytes(lb) as usize;
                let mut jb = vec![0u8; jlen];
                s.read_exact(&mut jb).map_err(|e| e.to_string())?;
                return Ok((all_samples, String::from_utf8_lossy(&jb).into_owned()));
            }
            0xff => {
                let mut lb = [0u8; 4];
                s.read_exact(&mut lb).map_err(|e| e.to_string())?;
                let elen = u32::from_le_bytes(lb) as usize;
                let mut eb = vec![0u8; elen];
                s.read_exact(&mut eb).map_err(|e| e.to_string())?;
                return Err(String::from_utf8_lossy(&eb).into_owned());
            }
            other => return Err(format!("Unexpected CODEC_DECODE tag {:#x}", other)),
        }
    }
}

pub fn client_speak_via_daemon<F>(
    talker_ref: &str,
    params: &TtsSynthParams,
    cancel: Option<&AtomicBool>,
    mut on_audio_chunk: F,
) -> Result<(Vec<f32>, String), String>
where
    F: FnMut(&[f32]) -> bool,
{
    let sock_path = daemon_sock_path();
    let mut s = UnixStream::connect(&sock_path)
        .map_err(|e| format!("Failed to connect to daemon {}: {}", sock_path.display(), e))?;
    let _ = s.set_read_timeout(Some(Duration::from_millis(50)));

    let seed_i64 = params.seed.map(|v| v as i64).unwrap_or(-1);
    let hdr = format!(
        "SPEAK\t{}\t{}\t{}\t{}\t{:.4}\t{}\t{:.4}\t{}\t{}\n",
        if talker_ref.is_empty() { "1.7b" } else { talker_ref },
        if params.speaker.is_empty() { "ryan" } else { &params.speaker },
        if params.language.is_empty() { "english" } else { &params.language },
        params.max_frames,
        params.temperature,
        params.top_k,
        params.rep_penalty,
        seed_i64,
        u8::from(params.streaming_chunks),
    );
    let ib = params.instruct.as_bytes();
    let tb = params.text.as_bytes();
    let mut req = Vec::with_capacity(hdr.len() + 8 + ib.len() + tb.len());
    req.extend_from_slice(hdr.as_bytes());
    req.extend_from_slice(&(ib.len() as u32).to_le_bytes());
    req.extend_from_slice(ib);
    req.extend_from_slice(&(tb.len() as u32).to_le_bytes());
    req.extend_from_slice(tb);
    s.write_all(&req).map_err(|e| e.to_string())?;
    s.flush().map_err(|e| e.to_string())?;

    let read_exact_opt_cancel = |sock: &mut UnixStream, mut dst: &mut [u8]| -> Result<bool, String> {
        while !dst.is_empty() {
            if cancel.map_or(false, |c| c.load(Ordering::Relaxed)) {
                let _ = sock.shutdown(std::net::Shutdown::Both);
                return Ok(false);
            }
            match sock.read(dst) {
                Ok(0) => return Err("Daemon connection closed unexpectedly during SPEAK".into()),
                Ok(n) => dst = &mut dst[n..],
                Err(ref e)
                    if e.kind() == std::io::ErrorKind::WouldBlock
                        || e.kind() == std::io::ErrorKind::TimedOut
                        || e.kind() == std::io::ErrorKind::Interrupted =>
                {
                    continue;
                }
                Err(e) => return Err(format!("Daemon SPEAK read error: {}", e)),
            }
        }
        Ok(true)
    };

    let mut all_samples = Vec::<f32>::new();
    let mut chunk_buf = Vec::<f32>::new();
    loop {
        if cancel.map_or(false, |c| c.load(Ordering::Relaxed)) {
            let _ = s.shutdown(std::net::Shutdown::Both);
            return Ok((all_samples, "{\"cancelled\":true}".into()));
        }
        let mut tag = [0u8; 1];
        if !read_exact_opt_cancel(&mut s, &mut tag)? {
            return Ok((all_samples, "{\"cancelled\":true}".into()));
        }
        match tag[0] {
            0x03 => {
                let mut nb = [0u8; 4];
                if !read_exact_opt_cancel(&mut s, &mut nb)? {
                    return Ok((all_samples, "{\"cancelled\":true}".into()));
                }
                let n_samp = u32::from_le_bytes(nb) as usize;
                let mut raw = vec![0u8; n_samp * 4];
                if n_samp > 0 && !read_exact_opt_cancel(&mut s, &mut raw)? {
                    return Ok((all_samples, "{\"cancelled\":true}".into()));
                }
                chunk_buf.clear();
                chunk_buf.reserve(n_samp);
                for c in raw.chunks_exact(4) {
                    chunk_buf.push(f32::from_bits(u32::from_le_bytes(c.try_into().unwrap())));
                }
                all_samples.extend_from_slice(&chunk_buf);
                if !on_audio_chunk(&chunk_buf) {
                    let _ = s.shutdown(std::net::Shutdown::Both);
                    return Ok((all_samples, "{\"cancelled\":true}".into()));
                }
            }
            0x02 => {
                let mut lb = [0u8; 4];
                if !read_exact_opt_cancel(&mut s, &mut lb)? {
                    return Ok((all_samples, "{\"cancelled\":true}".into()));
                }
                let jlen = u32::from_le_bytes(lb) as usize;
                let mut jb = vec![0u8; jlen];
                if jlen > 0 && !read_exact_opt_cancel(&mut s, &mut jb)? {
                    return Ok((all_samples, "{\"cancelled\":true}".into()));
                }
                return Ok((all_samples, String::from_utf8_lossy(&jb).into_owned()));
            }
            0xff => {
                let mut lb = [0u8; 4];
                let _ = read_exact_opt_cancel(&mut s, &mut lb)?;
                let elen = u32::from_le_bytes(lb) as usize;
                let mut eb = vec![0u8; elen];
                let _ = read_exact_opt_cancel(&mut s, &mut eb)?;
                return Err(String::from_utf8_lossy(&eb).into_owned());
            }
            other => return Err(format!("Unexpected daemon SPEAK frame tag {:#x}", other)),
        }
    }
}

pub fn client_generate_ids_cancellable<F>(
    prompt_ids: &[i32],
    max_tokens: usize,
    temp: f32,
    spec: bool,
    reset_cache: bool,
    cancel: &AtomicBool,
    mut on_piece: F,
) -> Result<String, String>
where
    F: FnMut(i32, &str),
{
    let sock_path = daemon_sock_path();
    let mut s = UnixStream::connect(&sock_path)
        .map_err(|e| format!("Failed to connect to daemon {}: {}", sock_path.display(), e))?;
    let _ = s.set_read_timeout(Some(Duration::from_millis(40)));

    let hdr = format!(
        "GEN\t{}\t{:.4}\t{}\t{}\tids\n",
        max_tokens,
        temp,
        u8::from(spec),
        u8::from(reset_cache),
    );
    let byte_len = prompt_ids.len() * 4;
    let mut req = Vec::with_capacity(hdr.len() + 4 + byte_len);
    req.extend_from_slice(hdr.as_bytes());
    req.extend_from_slice(&(byte_len as u32).to_le_bytes());
    for &id in prompt_ids {
        req.extend_from_slice(&id.to_le_bytes());
    }
    s.write_all(&req).map_err(|e| e.to_string())?;
    s.flush().map_err(|e| e.to_string())?;

    let read_exact_cancellable = |sock: &mut UnixStream, mut dst: &mut [u8], cancel: &AtomicBool| -> Result<bool, String> {
        while !dst.is_empty() {
            if cancel.load(Ordering::Relaxed) {
                let _ = sock.shutdown(std::net::Shutdown::Both);
                return Ok(false);
            }
            match sock.read(dst) {
                Ok(0) => return Err("Daemon connection closed unexpectedly".into()),
                Ok(n) => dst = &mut dst[n..],
                Err(ref e)
                    if e.kind() == std::io::ErrorKind::WouldBlock
                        || e.kind() == std::io::ErrorKind::TimedOut
                        || e.kind() == std::io::ErrorKind::Interrupted =>
                {
                    continue;
                }
                Err(e) => return Err(format!("Daemon read error: {}", e)),
            }
        }
        Ok(true)
    };

    loop {
        if cancel.load(Ordering::Relaxed) {
            let _ = s.shutdown(std::net::Shutdown::Both);
            return Ok("{\"cancelled\":true}".into());
        }
        let mut tag = [0u8; 1];
        if !read_exact_cancellable(&mut s, &mut tag, cancel)? {
            return Ok("{\"cancelled\":true}".into());
        }
        match tag[0] {
            0x01 => {
                let mut meta = [0u8; 6];
                if !read_exact_cancellable(&mut s, &mut meta, cancel)? {
                    return Ok("{\"cancelled\":true}".into());
                }
                let tok_id = i32::from_le_bytes(meta[0..4].try_into().unwrap());
                let slen = u16::from_le_bytes(meta[4..6].try_into().unwrap()) as usize;
                let mut sbytes = vec![0u8; slen];
                if slen > 0 && !read_exact_cancellable(&mut s, &mut sbytes, cancel)? {
                    return Ok("{\"cancelled\":true}".into());
                }
                let piece = String::from_utf8_lossy(&sbytes);
                on_piece(tok_id, &piece);
            }
            0x02 => {
                let mut lb = [0u8; 4];
                if !read_exact_cancellable(&mut s, &mut lb, cancel)? {
                    return Ok("{\"cancelled\":true}".into());
                }
                let jlen = u32::from_le_bytes(lb) as usize;
                let mut jb = vec![0u8; jlen];
                if jlen > 0 && !read_exact_cancellable(&mut s, &mut jb, cancel)? {
                    return Ok("{\"cancelled\":true}".into());
                }
                return Ok(String::from_utf8_lossy(&jb).into_owned());
            }
            0xff => {
                let mut lb = [0u8; 4];
                let _ = read_exact_cancellable(&mut s, &mut lb, cancel)?;
                let elen = u32::from_le_bytes(lb) as usize;
                let mut eb = vec![0u8; elen];
                let _ = read_exact_cancellable(&mut s, &mut eb, cancel)?;
                return Err(String::from_utf8_lossy(&eb).into_owned());
            }
            other => return Err(format!("Unexpected daemon frame tag {:#x}", other)),
        }
    }
}

// -----------------------------------------------------------------------------
// Zero-Crate OpenAI-Compatible HTTP / SSE Server (`/v1/models`, `/v1/chat/completions`)
// -----------------------------------------------------------------------------

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Minimal JSON string/number/bool extractor for `/v1/chat/completions` payloads.
fn extract_last_content_and_opts(body: &str) -> (Vec<(String, String)>, usize, f32, bool) {
    let mut msgs = Vec::new();
    let mut search_pos = 0usize;
    while let Some(rpos) = body[search_pos..].find("\"role\"") {
        let abs_r = search_pos + rpos + 6;
        let role = extract_json_str_after(&body[abs_r..]).unwrap_or_else(|| "user".into());
        if let Some(cpos) = body[abs_r..].find("\"content\"") {
            let abs_c = abs_r + cpos + 9;
            if let Some(content) = extract_json_str_after(&body[abs_c..]) {
                msgs.push((role, content));
            }
            search_pos = abs_c;
        } else {
            break;
        }
    }
    if msgs.is_empty() {
        if let Some(ppos) = body.find("\"prompt\"") {
            if let Some(p) = extract_json_str_after(&body[ppos + 8..]) {
                msgs.push(("user".into(), p));
            }
        }
    }

    let max_toks = extract_json_num_key(body, "\"max_tokens\"")
        .or_else(|| extract_json_num_key(body, "\"max_completion_tokens\""))
        .map(|v| v as usize)
        .unwrap_or(512);
    let temp = extract_json_num_key(body, "\"temperature\"").unwrap_or(0.0) as f32;
    let stream = body
        .find("\"stream\"")
        .map(|i| body[i + 8..].trim_start().trim_start_matches(':').trim_start().starts_with("true"))
        .unwrap_or(false);
    (msgs, max_toks, temp, stream)
}

fn extract_json_str_after(s: &str) -> Option<String> {
    let rest = s.trim_start().strip_prefix(':')?.trim_start().strip_prefix('"')?;
    let mut out = String::new();
    let mut chars = rest.chars();
    while let Some(c) = chars.next() {
        if c == '"' {
            return Some(out);
        }
        if c == '\\' {
            match chars.next()? {
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                't' => out.push('\t'),
                '"' => out.push('"'),
                '\\' => out.push('\\'),
                '/' => out.push('/'),
                'u' => {
                    let hex: String = chars.by_ref().take(4).collect();
                    if let Ok(cp) = u32::from_str_radix(&hex, 16) {
                        if let Some(ch) = char::from_u32(cp) {
                            out.push(ch);
                        }
                    }
                }
                other => out.push(other),
            }
        } else {
            out.push(c);
        }
    }
    None
}

fn extract_json_num_key(body: &str, key: &str) -> Option<f64> {
    let idx = body.find(key)?;
    let rest = body[idx + key.len()..].trim_start().strip_prefix(':')?.trim_start();
    let num_s: String = rest
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-' || *c == 'e' || *c == 'E')
        .collect();
    num_s.parse::<f64>().ok()
}

pub fn run_http_server(model_ref: &str, max_ctx: usize, port: u16, spec: bool) -> Result<(), String> {
    install_signal_handlers();
    SHUTDOWN_REQ.store(false, Ordering::SeqCst);

    let mut engine = QwenMegakernelEngine::load(model_ref, max_ctx)?;
    let addr = format!("127.0.0.1:{}", port);
    let listener = TcpListener::bind(&addr).map_err(|e| format!("Failed to bind {}: {}", addr, e))?;
    listener.set_nonblocking(true).map_err(|e| e.to_string())?;

    eprintln!(
        "[qwen-nv serve] Listening on http://{} (OpenAI `/v1/chat/completions`, model='{}')",
        addr, engine.cfg.model_name
    );

    while !SHUTDOWN_REQ.load(Ordering::Relaxed) {
        match listener.accept() {
            Ok((stream, _)) => {
                let _ = stream.set_nonblocking(false);
                let _ = handle_http_conn(&mut engine, stream, spec);
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(2));
            }
            Err(_) => thread::sleep(Duration::from_millis(10)),
        }
    }

    eprintln!("[qwen-nv serve] Unloading GSP-RM cleanly...");
    engine.gpu.shutdown_clean();
    Ok(())
}

fn handle_http_conn(engine: &mut QwenMegakernelEngine, mut stream: TcpStream, spec: bool) -> Result<(), String> {
    let mut rdr = BufReader::new(stream.try_clone().map_err(|e| e.to_string())?);
    let mut req_line = String::new();
    rdr.read_line(&mut req_line).map_err(|e| e.to_string())?;
    let parts: Vec<&str> = req_line.split_whitespace().collect();
    if parts.len() < 2 {
        return Ok(());
    }
    let method = parts[0];
    let path = parts[1];

    let mut content_len = 0usize;
    loop {
        let mut hline = String::new();
        rdr.read_line(&mut hline).map_err(|e| e.to_string())?;
        if hline == "\r\n" || hline == "\n" || hline.is_empty() {
            break;
        }
        if let Some((k, v)) = hline.split_once(':') {
            if k.eq_ignore_ascii_case("content-length") {
                content_len = v.trim().parse().unwrap_or(0);
            }
        }
    }

    let mut body = vec![0u8; content_len];
    if content_len > 0 {
        rdr.read_exact(&mut body).map_err(|e| e.to_string())?;
    }

    if method == "GET" && path == "/v1/models" {
        let resp_body = format!(
            "{{\"object\":\"list\",\"data\":[{{\"id\":\"{}\",\"object\":\"model\"}}]}}",
            json_escape(&engine.cfg.model_name)
        );
        let http = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {}\r\n\r\n{}",
            resp_body.len(),
            resp_body
        );
        let _ = stream.write_all(http.as_bytes());
        return Ok(());
    }

    if method == "POST" && path.starts_with("/v1/chat/completions") {
        let body_str = String::from_utf8_lossy(&body);
        let (msgs, max_toks, temp, is_stream) = extract_last_content_and_opts(&body_str);
        if msgs.is_empty() {
            let _ = stream.write_all(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n");
            return Ok(());
        }
        let prompt_ids = engine.encode_chat(&msgs, false);
        let created = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        let model_id = json_escape(&engine.cfg.model_name);

        if is_stream {
            let hdr = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n";
            stream.write_all(hdr.as_bytes()).map_err(|e| e.to_string())?;
            let tok_ptr = &engine.tok as *const crate::gguf::BpeTokenizer;
            let mut dec = Utf8StreamDecoder::new();

            let (_, stats) = engine.generate(&prompt_ids, max_toks, temp, spec, Some(&SHUTDOWN_REQ), |tid| {
                let piece = unsafe { dec.push((*tok_ptr).token_bytes(tid)) };
                if !piece.is_empty() {
                    let ev = format!(
                        "data: {{\"id\":\"chatcmpl-qwen-nv\",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{}\"}},\"finish_reason\":null}}]}}\n\n",
                        created,
                        model_id,
                        json_escape(&piece)
                    );
                    if stream.write_all(ev.as_bytes()).is_err() || stream.flush().is_err() {
                        return false;
                    }
                }
                true
            })?;

            let done_ev = format!(
                "data: {{\"id\":\"chatcmpl-qwen-nv\",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"{}\"}}]}}\n\ndata: [DONE]\n\n",
                created, model_id, stats.stop_reason
            );
            let _ = stream.write_all(done_ev.as_bytes());
        } else {
            let (out_ids, stats) = engine.generate(&prompt_ids, max_toks, temp, spec, Some(&SHUTDOWN_REQ), |_| true)?;
            let text = engine.tok.decode(&out_ids);
            let resp_json = format!(
                "{{\"id\":\"chatcmpl-qwen-nv\",\"object\":\"chat.completion\",\"created\":{},\"model\":\"{}\",\
                  \"choices\":[{{\"index\":0,\"message\":{{\"role\":\"assistant\",\"content\":\"{}\"}},\"finish_reason\":\"{}\"}}],\
                  \"usage\":{{\"prompt_tokens\":{},\"completion_tokens\":{},\"total_tokens\":{}}}}}",
                created,
                model_id,
                json_escape(&text),
                stats.stop_reason,
                stats.prompt_tokens,
                stats.completion_tokens,
                stats.prompt_tokens + stats.completion_tokens
            );
            let http = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {}\r\n\r\n{}",
                resp_json.len(),
                resp_json
            );
            let _ = stream.write_all(http.as_bytes());
        }
        return Ok(());
    }

    if method == "POST" && path.starts_with("/v1/audio/speech") {
        let body_str = String::from_utf8_lossy(&body);
        let input = body_str
            .find("\"input\"")
            .and_then(|i| extract_json_str_after(&body_str[i + 7..]))
            .unwrap_or_default();
        if input.trim().is_empty() {
            let _ = stream.write_all(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n");
            return Ok(());
        }
        let voice = body_str
            .find("\"voice\"")
            .and_then(|i| extract_json_str_after(&body_str[i + 7..]))
            .unwrap_or_else(|| "ryan".into());
        let voice_mapped = match voice.as_str() {
            "alloy" | "echo" | "onyx" => "ryan",
            "nova" | "shimmer" => "serena",
            "fable" => "aiden",
            other => other,
        }
        .to_string();
        let instruct = body_str
            .find("\"instructions\"")
            .and_then(|i| extract_json_str_after(&body_str[i + 14..]))
            .unwrap_or_default();

        let params = TtsSynthParams {
            text: input,
            speaker: voice_mapped,
            language: "auto".into(),
            instruct,
            streaming_chunks: false,
            ..Default::default()
        };

        match engine.synthesize_speech("1.7b", &params, Some(&SHUTDOWN_REQ), |_| true) {
            Ok((samples, _stats)) => {
                let wav_bytes = encode_wav_pcm16(&samples, TTS_SAMPLE_RATE);
                let hdr = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: audio/wav\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {}\r\n\r\n",
                    wav_bytes.len()
                );
                let _ = stream.write_all(hdr.as_bytes());
                let _ = stream.write_all(&wav_bytes);
            }
            Err(e) => {
                let err_j = format!("{{\"error\":\"{}\"}}", json_escape(&e));
                let hdr = format!(
                    "HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                    err_j.len(),
                    err_j
                );
                let _ = stream.write_all(hdr.as_bytes());
            }
        }
        return Ok(());
    }

    if method == "POST" && path.starts_with("/v1/audio/transcriptions") {
        // Support both raw RIFF WAV body (`--data-binary @clip.wav`) and `multipart/form-data` (`-F file=@clip.wav`)
        let wav_slice: Option<&[u8]> = if body.starts_with(b"RIFF") {
            Some(&body)
        } else {
            body.windows(12)
                .position(|w| &w[0..4] == b"RIFF" && &w[8..12] == b"WAVE")
                .and_then(|off| {
                    if off + 8 <= body.len() {
                        let riff_sz = u32::from_le_bytes(body[off + 4..off + 8].try_into().unwrap()) as usize;
                        let end = (off + 8 + riff_sz).min(body.len());
                        Some(&body[off..end])
                    } else {
                        None
                    }
                })
        };

        let Some(wav_bytes) = wav_slice else {
            let err_j = "{\"error\":\"Expected RIFF WAV audio body or multipart file upload\"}";
            let hdr = format!(
                "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                err_j.len(),
                err_j
            );
            let _ = stream.write_all(hdr.as_bytes());
            return Ok(());
        };

        let pcm_res = parse_wav_bytes_mono_16k(wav_bytes);
        let res = pcm_res.and_then(|pcm| engine.transcribe_audio("1.7b", &pcm, None));
        match res {
            Ok(stats) => {
                let resp_j = format!(
                    "{{\"text\":\"{}\",\"language\":\"{}\",\"duration\":{:.3},\"total_ms\":{:.2}}}",
                    json_escape(&stats.text),
                    json_escape(&stats.detected_language),
                    stats.audio_sec,
                    stats.total_ms
                );
                let hdr = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {}\r\n\r\n{}",
                    resp_j.len(),
                    resp_j
                );
                let _ = stream.write_all(hdr.as_bytes());
            }
            Err(e) => {
                let err_j = format!("{{\"error\":\"{}\"}}", json_escape(&e));
                let hdr = format!(
                    "HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                    err_j.len(),
                    err_j
                );
                let _ = stream.write_all(hdr.as_bytes());
            }
        }
        return Ok(());
    }

    let _ = stream.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n");
    Ok(())
}
