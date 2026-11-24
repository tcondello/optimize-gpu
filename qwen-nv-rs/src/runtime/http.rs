//! Zero-Crate OpenAI-Compatible HTTP / SSE Server (`/v1/models`, `/v1/chat/completions`,
//! `/v1/audio/speech`, `/v1/audio/transcriptions`, `/v1/embeddings`).

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::Ordering;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use super::session::RuntimeSession;
use super::{extract_json_str_after, install_signal_handlers_raw, json_escape, SHUTDOWN_REQ};
use crate::models::rag::EmbedStrategy;
use crate::voice::audio::{encode_wav_pcm16, parse_wav_bytes_mono_16k, TTS_SAMPLE_RATE};
use crate::voice::tts::TtsSynthParams;

fn extract_json_num_key(body: &str, key: &str) -> Option<f64> {
    let idx = body.find(key)?;
    let rest = body[idx + key.len()..].trim_start().strip_prefix(':')?.trim_start();
    let num_s: String = rest
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-' || *c == 'e' || *c == 'E')
        .collect();
    num_s.parse::<f64>().ok()
}

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
        .map(|i| {
            body[i + 8..]
                .trim_start()
                .trim_start_matches(':')
                .trim_start()
                .starts_with("true")
        })
        .unwrap_or(false);
    (msgs, max_toks, temp, stream)
}

pub fn run_http_server(model_ref: &str, max_ctx: usize, port: u16, spec: bool) -> Result<(), String> {
    install_signal_handlers_raw();
    SHUTDOWN_REQ.store(false, Ordering::SeqCst);

    let mut session = RuntimeSession::new(model_ref, max_ctx)?;
    let addr = format!("127.0.0.1:{}", port);
    let listener = TcpListener::bind(&addr).map_err(|e| format!("Failed to bind {}: {}", addr, e))?;
    listener.set_nonblocking(true).map_err(|e| e.to_string())?;

    eprintln!(
        "[qwen-nv serve] Listening on http://{} (OpenAI `/v1/chat/completions`, `/v1/audio/*`, model='{}')",
        addr,
        session.model_name()
    );

    let mut last_gpu_tick = Instant::now();
    while !SHUTDOWN_REQ.load(Ordering::Relaxed) {
        match listener.accept() {
            Ok((stream, _)) => {
                let _ = stream.set_nonblocking(false);
                let _ = handle_http_conn(&mut session, stream, spec);
                last_gpu_tick = Instant::now();
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                if last_gpu_tick.elapsed() >= Duration::from_secs(25) {
                    session.keepalive_or_recover();
                    last_gpu_tick = Instant::now();
                }
                thread::sleep(Duration::from_millis(2));
            }
            Err(_) => thread::sleep(Duration::from_millis(10)),
        }
    }

    eprintln!("[qwen-nv serve] Unloading GSP-RM cleanly...");
    session.shutdown_clean();
    Ok(())
}

fn handle_http_conn(session: &mut RuntimeSession, mut stream: TcpStream, spec: bool) -> Result<(), String> {
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
            json_escape(session.model_name())
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
        let created = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        let model_id = json_escape(session.model_name());

        if is_stream {
            let hdr = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n";
            stream.write_all(hdr.as_bytes()).map_err(|e| e.to_string())?;

            let (_, stop_reason, _, _) =
                session.http_chat_completion(&msgs, max_toks, temp, spec, Some(&SHUTDOWN_REQ), |piece| {
                    if !piece.is_empty() {
                        let ev = format!(
                            "data: {{\"id\":\"chatcmpl-qwen-nv\",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{}\"}},\"finish_reason\":null}}]}}\n\n",
                            created,
                            model_id,
                            json_escape(piece)
                        );
                        if stream.write_all(ev.as_bytes()).is_err() || stream.flush().is_err() {
                            return false;
                        }
                    }
                    true
                })?;

            let done_ev = format!(
                "data: {{\"id\":\"chatcmpl-qwen-nv\",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"{}\"}}]}}\n\ndata: [DONE]\n\n",
                created, model_id, stop_reason
            );
            let _ = stream.write_all(done_ev.as_bytes());
        } else {
            let (text, stop_reason, prompt_toks, comp_toks) =
                session.http_chat_completion(&msgs, max_toks, temp, spec, Some(&SHUTDOWN_REQ), |_| true)?;
            let resp_json = format!(
                "{{\"id\":\"chatcmpl-qwen-nv\",\"object\":\"chat.completion\",\"created\":{},\"model\":\"{}\",\
                  \"choices\":[{{\"index\":0,\"message\":{{\"role\":\"assistant\",\"content\":\"{}\"}},\"finish_reason\":\"{}\"}}],\
                  \"usage\":{{\"prompt_tokens\":{},\"completion_tokens\":{},\"total_tokens\":{}}}}}",
                created,
                model_id,
                json_escape(&text),
                stop_reason,
                prompt_toks,
                comp_toks,
                prompt_toks + comp_toks
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

        match session.synthesize_speech("default", &params, Some(&SHUTDOWN_REQ), |_| true) {
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
        let res = pcm_res.and_then(|pcm| session.transcribe_audio("default", &pcm, None));
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

    if method == "POST" && path.starts_with("/v1/embeddings") {
        let body_str = String::from_utf8_lossy(&body);
        let input = body_str
            .find("\"input\"")
            .and_then(|i| extract_json_str_after(&body_str[i + 7..]))
            .unwrap_or_default();
        if input.trim().is_empty() {
            let _ = stream.write_all(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n");
            return Ok(());
        }
        let strat_str = body_str
            .find("\"strategy\"")
            .and_then(|i| extract_json_str_after(&body_str[i + 10..]))
            .unwrap_or_else(|| "echo".into());
        let strat = EmbedStrategy::from_str(&strat_str).unwrap_or(EmbedStrategy::Echo);
        let model_name_esc = json_escape(session.model_name());
        match session.embed_text(strat, &input, true, None) {
            Ok((vec_f32, ms)) => {
                let nums: Vec<String> = vec_f32.iter().map(|v| format!("{:.6}", v)).collect();
                let resp_j = format!(
                    "{{\"object\":\"list\",\"data\":[{{\"object\":\"embedding\",\"index\":0,\"embedding\":[{}]}}],\"model\":\"{}\",\"strategy\":\"{}\",\"embed_ms\":{:.2}}}",
                    nums.join(","),
                    model_name_esc,
                    strat.as_str(),
                    ms
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
