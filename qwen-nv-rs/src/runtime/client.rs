//! Narrow Typed Unix-Socket Client Interface for the Persistent GPU Daemon.

#![allow(dead_code)]

use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::net::UnixStream;
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use super::{extract_json_str_after, install_signal_handlers_raw};
use crate::models::rag::{RagHit, RagSearchStats};
use crate::topic_lora::{json_find_num, json_find_str};
use crate::voice::tts::TtsSynthParams;

pub(crate) fn daemon_sock_path() -> PathBuf {
    if let Ok(p) = std::env::var("QWEN_NV_DAEMON_SOCK") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    PathBuf::from(home).join(".cache/qwen-nv-rs/daemon.sock")
}

fn send_simple_cmd(cmd_line: &str) -> Result<String, String> {
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
    if trimmed == "OK" {
        Ok("OK".to_string())
    } else if let Some(rest) = trimmed.strip_prefix("OK\t") {
        Ok(rest.to_string())
    } else if let Some(err) = trimmed.strip_prefix("ERR\t") {
        Err(err.to_string())
    } else {
        Err(format!("Unexpected daemon reply: {}", trimmed))
    }
}

pub enum DaemonQuery {
    ResetCache,
    LoraLoad { path: String, scale: f32 },
    LoraClear,
    LoraStatus,
    TtsWarm(String),
    AsrWarm(String),
    RagStatus,
    RagClear,
    RagLoad(String),
    DrafterStatus,
    DrafterClear,
    DrafterLoad { draft_ref: String, prior_path: String },
    TopicLoad(String),
    TopicStatus,
    TopicList,
    TopicClear,
    InklingSetGamma(usize),
}

pub struct DaemonControl;

impl DaemonControl {
    pub fn sock_path() -> PathBuf {
        daemon_sock_path()
    }

    pub fn install_signal_handlers() {
        install_signal_handlers_raw();
    }

    pub fn ping() -> Result<(String, String, u32), String> {
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

    pub fn stop() -> Result<bool, String> {
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

    pub fn ensure_running(model_ref: &str, max_ctx: usize) -> Result<(), String> {
        if let Ok((loaded_ref, _, _)) = Self::ping() {
            if loaded_ref == model_ref {
                return Ok(());
            }
            eprintln!(
                "[qwen-nv] Daemon currently holds '{}'; restarting for '{}'...",
                loaded_ref, model_ref
            );
            let _ = Self::stop();
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
        while t0.elapsed() < Duration::from_secs(60) {
            if Self::ping().is_ok() {
                return Ok(());
            }
            if let Ok(Some(status)) = child.try_wait() {
                let log_tail = fs::read_to_string(&log_path).unwrap_or_default();
                return Err(format!("Daemon exited early ({}):\n{}", status, log_tail));
            }
            thread::sleep(Duration::from_millis(50));
        }
        Err("Timed out waiting 60s for background GPU daemon to initialize".into())
    }

    pub fn query(q: DaemonQuery) -> Result<String, String> {
        let cmd = match q {
            DaemonQuery::ResetCache => "RESET\n".to_string(),
            DaemonQuery::LoraLoad { path, scale } => format!("LORA_LOAD\t{:.4}\t{}\n", scale, path),
            DaemonQuery::LoraClear => "LORA_CLEAR\n".to_string(),
            DaemonQuery::LoraStatus => "LORA_STATUS\n".to_string(),
            DaemonQuery::TtsWarm(r) => format!("TTS_WARM\t{}\n", r),
            DaemonQuery::AsrWarm(r) => format!("ASR_WARM\t{}\n", r),
            DaemonQuery::RagStatus => "RAG_STATUS\n".to_string(),
            DaemonQuery::RagClear => "RAG_CLEAR\n".to_string(),
            DaemonQuery::RagLoad(p) => format!("RAG_LOAD\t{}\n", p),
            DaemonQuery::DrafterStatus => "DRAFTER_STATUS\n".to_string(),
            DaemonQuery::DrafterClear => "DRAFTER_CLEAR\n".to_string(),
            DaemonQuery::DrafterLoad { draft_ref, prior_path } => {
                format!("DRAFTER_LOAD\t{}\t{}\n", draft_ref, prior_path)
            }
            DaemonQuery::TopicLoad(p) => format!("TOPIC_LOAD\t{}\n", p),
            DaemonQuery::TopicStatus => "TOPIC_STATUS\n".to_string(),
            DaemonQuery::TopicList => "TOPIC_LIST\n".to_string(),
            DaemonQuery::TopicClear => "TOPIC_CLEAR\n".to_string(),
            DaemonQuery::InklingSetGamma(g) => format!("INKLING_CFG\t{}\n", g.clamp(1, 7)),
        };
        send_simple_cmd(&cmd)
    }
}

fn generate_via_daemon_raw<F>(
    payload: &[u8],
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

    let co_override = std::env::var("QWEN_NV_SPEC_CROSSOVER").unwrap_or_default();
    let hdr = format!(
        "GEN\t{}\t{:.4}\t{}\t{}\t{}\t{}\n",
        max_tokens,
        temp,
        u8::from(spec),
        u8::from(reset_cache),
        mode,
        co_override
    );
    let mut req = Vec::with_capacity(hdr.len() + 4 + payload.len());
    req.extend_from_slice(hdr.as_bytes());
    req.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    req.extend_from_slice(payload);
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

pub struct DaemonStream;

impl DaemonStream {
    pub fn generate<F>(
        prompt: &str,
        mode: &str,
        max_tokens: usize,
        temp: f32,
        spec: bool,
        reset_cache: bool,
        on_piece: F,
    ) -> Result<String, String>
    where
        F: FnMut(i32, &str),
    {
        generate_via_daemon_raw(prompt.as_bytes(), mode, max_tokens, temp, spec, reset_cache, on_piece)
    }

    pub fn generate_chat_sys<F>(
        system_prompt: &str,
        user_prompt: &str,
        max_tokens: usize,
        temp: f32,
        spec: bool,
        reset_cache: bool,
        on_piece: F,
    ) -> Result<String, String>
    where
        F: FnMut(i32, &str),
    {
        let sb = system_prompt.as_bytes();
        let ub = user_prompt.as_bytes();
        let mut payload = Vec::with_capacity(4 + sb.len() + ub.len());
        payload.extend_from_slice(&(sb.len() as u32).to_le_bytes());
        payload.extend_from_slice(sb);
        payload.extend_from_slice(ub);
        generate_via_daemon_raw(&payload, "chat_sys", max_tokens, temp, spec, reset_cache, on_piece)
    }

    pub fn generate_ids_cancellable<F>(
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

        let read_exact_cancellable =
            |sock: &mut UnixStream, mut dst: &mut [u8], cancel: &AtomicBool| -> Result<bool, String> {
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

    pub fn transcribe(asr_ref: &str, lang_hint: &str, pcm_16k: &[f32]) -> Result<(String, String), String> {
        let sock_path = daemon_sock_path();
        let mut s = UnixStream::connect(&sock_path)
            .map_err(|e| format!("Failed to connect to daemon {}: {}", sock_path.display(), e))?;
        let _ = s.set_read_timeout(Some(Duration::from_secs(60)));

        let eff_ref = if asr_ref.trim().is_empty() { "default" } else { asr_ref.trim() };
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

    pub fn codec_decode(talker_ref: &str, codes_flat: &[i32], n_frames: usize) -> Result<(Vec<f32>, String), String> {
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

    pub fn speak<F>(
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
            if talker_ref.is_empty() { "default" } else { talker_ref },
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
}

#[derive(Clone, Debug)]
pub struct TopicRouteClientReply {
    pub matched: bool,
    pub topic: String,
    pub slug: String,
    pub score: f32,
    pub lora_dir: String,
    pub system_prompt: String,
    pub embed_ms: f64,
    pub gpu_search_us: f64,
    pub swap_ms: f64,
    pub raw_json: String,
}

pub struct DaemonRagClient;

impl DaemonRagClient {
    pub fn embed(
        strategy: &str,
        is_query: bool,
        lora_path: &str,
        text: &str,
    ) -> Result<(Vec<f32>, f64), String> {
        let sock_path = daemon_sock_path();
        let mut s = UnixStream::connect(&sock_path)
            .map_err(|e| format!("Failed to connect to daemon {}: {}", sock_path.display(), e))?;
        let _ = s.set_read_timeout(Some(Duration::from_secs(60)));
        let tb = text.as_bytes();
        let hdr = format!(
            "EMBED\t{}\t{}\t{}\t{}\n",
            strategy,
            u8::from(is_query),
            tb.len(),
            lora_path
        );
        let mut req = Vec::with_capacity(hdr.len() + tb.len());
        req.extend_from_slice(hdr.as_bytes());
        req.extend_from_slice(tb);
        s.write_all(&req).map_err(|e| e.to_string())?;
        s.flush().map_err(|e| e.to_string())?;

        let mut rdr = BufReader::new(s);
        let mut line = String::new();
        rdr.read_line(&mut line).map_err(|e| e.to_string())?;
        let trimmed = line.trim_end_matches(&['\r', '\n'][..]);
        if let Some(err) = trimmed.strip_prefix("ERR\t") {
            return Err(err.to_string());
        }
        let rest = trimmed
            .strip_prefix("OK\t")
            .ok_or_else(|| format!("Unexpected EMBED reply: {}", trimmed))?;
        let parts: Vec<&str> = rest.split('\t').collect();
        if parts.len() < 2 {
            return Err(format!("Malformed EMBED reply: {}", trimmed));
        }
        let ms: f64 = parts[0].parse().unwrap_or(0.0);
        let dim: usize = parts[1].parse().unwrap_or(0);
        let mut raw = vec![0u8; dim * 4];
        rdr.read_exact(&mut raw).map_err(|e| e.to_string())?;
        let vec_f32: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_bits(u32::from_le_bytes(c.try_into().unwrap())))
            .collect();
        Ok((vec_f32, ms))
    }

    pub fn rag_index(
        strategy: &str,
        jsonl_path: &str,
        out_qvec_path: &str,
        lora_path: &str,
        load_into_vram: bool,
    ) -> Result<String, String> {
        let sock_path = daemon_sock_path();
        let mut s = UnixStream::connect(&sock_path)
            .map_err(|e| format!("Failed to connect to daemon {}: {}", sock_path.display(), e))?;
        let _ = s.set_read_timeout(Some(Duration::from_secs(2400)));
        let cmd = format!(
            "RAG_INDEX\t{}\t{}\t{}\t{}\t{}\n",
            strategy,
            u8::from(load_into_vram),
            jsonl_path,
            out_qvec_path,
            lora_path
        );
        s.write_all(cmd.as_bytes()).map_err(|e| e.to_string())?;
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
            Err(format!("Unexpected RAG_INDEX reply: {}", trimmed))
        }
    }

    pub fn rag_search(query: &str, top_k: usize) -> Result<RagSearchStats, String> {
        let sock_path = daemon_sock_path();
        let mut s = UnixStream::connect(&sock_path)
            .map_err(|e| format!("Failed to connect to daemon {}: {}", sock_path.display(), e))?;
        let _ = s.set_read_timeout(Some(Duration::from_secs(60)));
        let qb = query.as_bytes();
        let hdr = format!("RAG_SEARCH\t{}\t{}\n", top_k, qb.len());
        let mut req = Vec::with_capacity(hdr.len() + qb.len());
        req.extend_from_slice(hdr.as_bytes());
        req.extend_from_slice(qb);
        s.write_all(&req).map_err(|e| e.to_string())?;
        s.flush().map_err(|e| e.to_string())?;

        let mut rdr = BufReader::new(s);
        let mut line = String::new();
        rdr.read_line(&mut line).map_err(|e| e.to_string())?;
        let trimmed = line.trim_end_matches(&['\r', '\n'][..]);
        if let Some(err) = trimmed.strip_prefix("ERR\t") {
            return Err(err.to_string());
        }
        let rest = trimmed
            .strip_prefix("OK\t")
            .ok_or_else(|| format!("Unexpected RAG_SEARCH reply: {}", trimmed))?;
        let parts: Vec<&str> = rest.split('\t').collect();
        if parts.len() < 6 {
            return Err(format!("Malformed RAG_SEARCH header: {}", trimmed));
        }
        let hit_cnt: usize = parts[0].parse().unwrap_or(0);
        let embed_ms: f64 = parts[1].parse().unwrap_or(0.0);
        let gpu_search_us: f64 = parts[2].parse().unwrap_or(0.0);
        let total_ms: f64 = parts[3].parse().unwrap_or(0.0);
        let num_docs: usize = parts[4].parse().unwrap_or(0);
        let strategy = parts[5].to_string();

        let mut hits = Vec::with_capacity(hit_cnt);
        for _ in 0..hit_cnt {
            let mut meta = [0u8; 12];
            rdr.read_exact(&mut meta).map_err(|e| e.to_string())?;
            let id = u32::from_le_bytes(meta[0..4].try_into().unwrap());
            let score = f32::from_bits(u32::from_le_bytes(meta[4..8].try_into().unwrap()));
            let tlen = u32::from_le_bytes(meta[8..12].try_into().unwrap()) as usize;
            let mut tbuf = vec![0u8; tlen];
            if tlen > 0 {
                rdr.read_exact(&mut tbuf).map_err(|e| e.to_string())?;
            }
            let mut xlen_b = [0u8; 4];
            rdr.read_exact(&mut xlen_b).map_err(|e| e.to_string())?;
            let xlen = u32::from_le_bytes(xlen_b) as usize;
            let mut xbuf = vec![0u8; xlen];
            if xlen > 0 {
                rdr.read_exact(&mut xbuf).map_err(|e| e.to_string())?;
            }
            hits.push(RagHit {
                id,
                title: String::from_utf8_lossy(&tbuf).into_owned(),
                text: String::from_utf8_lossy(&xbuf).into_owned(),
                score,
            });
        }

        Ok(RagSearchStats {
            hits,
            embed_ms,
            gpu_search_us,
            total_ms,
            strategy,
            num_docs,
        })
    }

    pub fn topic_route(
        query: &str,
        top_k: usize,
        min_score: f32,
        auto_load: bool,
    ) -> Result<TopicRouteClientReply, String> {
        let sock_path = daemon_sock_path();
        let mut s = UnixStream::connect(&sock_path)
            .map_err(|e| format!("Failed to connect to daemon {}: {}", sock_path.display(), e))?;
        let _ = s.set_read_timeout(Some(Duration::from_secs(60)));
        let qb = query.as_bytes();
        let hdr = format!(
            "TOPIC_ROUTE\t{}\t{}\t{:.4}\t{}\n",
            u8::from(auto_load),
            top_k,
            min_score,
            qb.len()
        );
        let mut req = Vec::with_capacity(hdr.len() + qb.len());
        req.extend_from_slice(hdr.as_bytes());
        req.extend_from_slice(qb);
        s.write_all(&req).map_err(|e| e.to_string())?;
        s.flush().map_err(|e| e.to_string())?;

        let mut rdr = BufReader::new(s);
        let mut line = String::new();
        rdr.read_line(&mut line).map_err(|e| e.to_string())?;
        let trimmed = line.trim_end_matches(&['\r', '\n'][..]);
        if let Some(err) = trimmed.strip_prefix("ERR\t") {
            return Err(err.to_string());
        }
        let j = trimmed
            .strip_prefix("OK\t")
            .ok_or_else(|| format!("Unexpected TOPIC_ROUTE reply: {}", trimmed))?
            .to_string();

        Ok(TopicRouteClientReply {
            matched: j.contains("\"matched\":true"),
            topic: json_find_str(&j, "topic").unwrap_or_default(),
            slug: json_find_str(&j, "slug").unwrap_or_default(),
            score: json_find_num(&j, "score").unwrap_or(0.0) as f32,
            lora_dir: json_find_str(&j, "lora_dir").unwrap_or_default(),
            system_prompt: json_find_str(&j, "system_prompt").unwrap_or_default(),
            embed_ms: json_find_num(&j, "embed_ms").unwrap_or(0.0),
            gpu_search_us: json_find_num(&j, "gpu_search_us").unwrap_or(0.0),
            swap_ms: json_find_num(&j, "swap_ms").unwrap_or(0.0),
            raw_json: j,
        })
    }
}

fn vec_f32_cmd(cmd_prefix: &str, tok_ids: &[i32]) -> Result<Vec<f32>, String> {
    let sock_path = daemon_sock_path();
    let mut s = UnixStream::connect(&sock_path)
        .map_err(|e| format!("Failed to connect to daemon {}: {}", sock_path.display(), e))?;
    let _ = s.set_read_timeout(Some(Duration::from_secs(120)));
    let hdr = format!("{}\t{}\n", cmd_prefix, tok_ids.len());
    let mut req = Vec::with_capacity(hdr.len() + tok_ids.len() * 4);
    req.extend_from_slice(hdr.as_bytes());
    for &t in tok_ids {
        req.extend_from_slice(&t.to_le_bytes());
    }
    s.write_all(&req).map_err(|e| e.to_string())?;
    s.flush().map_err(|e| e.to_string())?;

    let mut rdr = BufReader::new(s);
    let mut line = String::new();
    rdr.read_line(&mut line).map_err(|e| e.to_string())?;
    let trimmed = line.trim_end_matches(&['\r', '\n'][..]);
    if let Some(err) = trimmed.strip_prefix("ERR\t") {
        return Err(err.to_string());
    }
    let rest = trimmed
        .strip_prefix("OK\t")
        .ok_or_else(|| format!("Unexpected {} reply: {}", cmd_prefix, trimmed))?;
    let n_floats: usize = rest.trim().parse().unwrap_or(0);
    let mut raw = vec![0u8; n_floats * 4];
    rdr.read_exact(&mut raw).map_err(|e| e.to_string())?;
    Ok(raw
        .chunks_exact(4)
        .map(|c| f32::from_bits(u32::from_le_bytes(c.try_into().unwrap())))
        .collect())
}

pub struct DaemonHooksClient;

impl DaemonHooksClient {
    pub fn extract_hooks(tok_ids: &[i32]) -> Result<Vec<f32>, String> {
        vec_f32_cmd("HOOKS_EXTRACT", tok_ids)
    }

    pub fn eval_logits(tok_ids: &[i32]) -> Result<Vec<f32>, String> {
        vec_f32_cmd("LOGITS_EVAL", tok_ids)
    }

    pub fn extract_mlp_kv(
        tok_ids: &[i32],
        layers: &[usize],
        pos_start: usize,
        pos_count: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let sock_path = daemon_sock_path();
        let mut s = UnixStream::connect(&sock_path)
            .map_err(|e| format!("Failed to connect to daemon {}: {}", sock_path.display(), e))?;
        let _ = s.set_read_timeout(Some(Duration::from_secs(120)));
        let lcsv: Vec<String> = layers.iter().map(|l| l.to_string()).collect();
        let hdr = format!(
            "MLP_KV_EXTRACT\t{}\t{}\t{}\t{}\n",
            tok_ids.len(),
            pos_start,
            pos_count,
            lcsv.join(",")
        );
        let mut req = Vec::with_capacity(hdr.len() + tok_ids.len() * 4);
        req.extend_from_slice(hdr.as_bytes());
        for &t in tok_ids {
            req.extend_from_slice(&t.to_le_bytes());
        }
        s.write_all(&req).map_err(|e| e.to_string())?;
        s.flush().map_err(|e| e.to_string())?;

        let mut rdr = BufReader::new(s);
        let mut line = String::new();
        rdr.read_line(&mut line).map_err(|e| e.to_string())?;
        let trimmed = line.trim_end_matches(&['\r', '\n'][..]);
        if let Some(err) = trimmed.strip_prefix("ERR\t") {
            return Err(err.to_string());
        }
        let rest = trimmed
            .strip_prefix("OK\t")
            .ok_or_else(|| format!("Unexpected MLP_KV_EXTRACT reply: {}", trimmed))?;
        let p: Vec<&str> = rest.split('\t').collect();
        if p.len() < 2 {
            return Err(format!("Malformed MLP_KV_EXTRACT header: {}", trimmed));
        }
        let k_len: usize = p[0].parse().unwrap_or(0);
        let r_len: usize = p[1].parse().unwrap_or(0);
        let mut raw_k = vec![0u8; k_len * 4];
        let mut raw_r = vec![0u8; r_len * 4];
        if !raw_k.is_empty() {
            rdr.read_exact(&mut raw_k).map_err(|e| e.to_string())?;
        }
        if !raw_r.is_empty() {
            rdr.read_exact(&mut raw_r).map_err(|e| e.to_string())?;
        }
        let keys = raw_k
            .chunks_exact(4)
            .map(|c| f32::from_bits(u32::from_le_bytes(c.try_into().unwrap())))
            .collect();
        let res = raw_r
            .chunks_exact(4)
            .map(|c| f32::from_bits(u32::from_le_bytes(c.try_into().unwrap())))
            .collect();
        Ok((keys, res))
    }
}
