//! Unified Unix-Domain-Socket Persistent GPU Daemon Server (`~/.cache/rtx/daemon.sock`).
//! Drives `RuntimeSession` so both `Qwen3` and `Inkling-2-Mini` share 100%-on-RTX-3080 STT and TTS.

use std::fs;
use std::io::{Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::sync::atomic::Ordering;
use std::thread;
use std::time::{Duration, Instant};

use super::client::{daemon_sock_path, DaemonControl};
use super::session::RuntimeSession;
use super::{install_signal_handlers_raw, json_escape, SHUTDOWN_REQ};
use crate::models::rag::{load_jsonl_corpus, EmbedStrategy};
use crate::voice::tts::TtsSynthParams;

pub fn run_daemon_foreground(model_ref: &str, max_ctx: usize) -> Result<(), String> {
    install_signal_handlers_raw();
    SHUTDOWN_REQ.store(false, Ordering::SeqCst);

    let sock_path = daemon_sock_path();
    if let Some(parent) = sock_path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    if sock_path.exists() {
        if DaemonControl::ping().is_ok() {
            return Err(format!("Daemon is already running at {}", sock_path.display()));
        }
        let _ = fs::remove_file(&sock_path);
    }

    let mut session = RuntimeSession::new(model_ref, max_ctx)?;
    session.warmup_for_daemon()?;

    let listener = UnixListener::bind(&sock_path)
        .map_err(|e| format!("Failed to bind {}: {}", sock_path.display(), e))?;
    listener.set_nonblocking(true).map_err(|e| e.to_string())?;

    let (_, model_name, active_sms) = session.ping_info();
    eprintln!(
        "[rtx daemon] Ready on {} (model='{}', SMs={}, warm latency < 1 ms)",
        sock_path.display(),
        model_name,
        active_sms
    );

    let mut last_gpu_tick = Instant::now();
    while !SHUTDOWN_REQ.load(Ordering::Relaxed) {
        match listener.accept() {
            Ok((stream, _)) => {
                let _ = stream.set_nonblocking(false);
                if let Ok(should_stop) = handle_daemon_client(&mut session, stream) {
                    if should_stop {
                        break;
                    }
                }
                last_gpu_tick = Instant::now();
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                if last_gpu_tick.elapsed() >= Duration::from_secs(25) {
                    session.keepalive_or_recover();
                    last_gpu_tick = Instant::now();
                }
                thread::sleep(Duration::from_millis(2));
            }
            Err(e) => {
                eprintln!("[rtx daemon] Accept error: {}", e);
                thread::sleep(Duration::from_millis(10));
            }
        }
    }

    eprintln!("[rtx daemon] Unloading GSP-RM cleanly...");
    let _ = fs::remove_file(&sock_path);
    session.shutdown_clean();
    Ok(())
}

fn write_reply_res(stream: &mut UnixStream, res: Result<String, String>) {
    match res {
        Ok(ok_payload) => {
            let _ = stream.write_all(format!("OK\t{}\n", ok_payload).as_bytes());
        }
        Err(e) => {
            let _ = stream.write_all(format!("ERR\t{}\n", e.replace('\n', " ")).as_bytes());
        }
    }
    let _ = stream.flush();
}

fn handle_daemon_client(session: &mut RuntimeSession, mut stream: UnixStream) -> Result<bool, String> {
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
            let (mref, mname, sms) = session.ping_info();
            let _ = stream.write_all(format!("PONG\t{}\t{}\t{}\n", mref, mname, sms).as_bytes());
            Ok(false)
        }
        "RESET" => {
            session.reset_cache();
            let _ = stream.write_all(b"OK\n");
            Ok(false)
        }
        "STOP" => {
            let _ = stream.write_all(b"BYE\n");
            Ok(true)
        }
        "INKLING_CFG" => {
            let g_opt = parts.get(1).and_then(|s| s.trim().parse::<usize>().ok());
            write_reply_res(&mut stream, Ok(session.inkling_set_gamma_json(g_opt)));
            Ok(false)
        }
        "LORA_STATUS" => {
            write_reply_res(&mut stream, Ok(session.lora_status_json()));
            Ok(false)
        }
        "LORA_CLEAR" => {
            let res = session.lora_clear_json();
            write_reply_res(&mut stream, res);
            Ok(false)
        }
        "LORA_LOAD" => {
            if parts.len() < 3 {
                write_reply_res(&mut stream, Err("Malformed LORA_LOAD".into()));
                return Ok(false);
            }
            let scale: f32 = parts[1].parse().unwrap_or(1.0);
            let path = parts[2..].join("\t");
            let res = session.lora_load_json(&path, scale);
            write_reply_res(&mut stream, res);
            Ok(false)
        }
        "TTS_WARM" => {
            let tk_ref = parts.get(1).copied().unwrap_or("default");
            let res = session
                .warm_tts(tk_ref)
                .map(|(mname, mb, spks)| format!("{}\t{:.1}\t{}", mname, mb, spks));
            write_reply_res(&mut stream, res);
            Ok(false)
        }
        "ASR_WARM" => {
            let asr_ref = parts.get(1).copied().unwrap_or("default");
            let res = session
                .warm_asr(asr_ref)
                .map(|(mname, mb)| format!("{}\t{:.1}", mname, mb));
            write_reply_res(&mut stream, res);
            Ok(false)
        }
        "TRANSCRIBE" => {
            if parts.len() < 4 {
                write_reply_res(&mut stream, Err("Malformed TRANSCRIBE header".into()));
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
            let res = session
                .transcribe_audio(asr_ref, &pcm_16k, lang_opt)
                .map(|st| st.to_json());
            write_reply_res(&mut stream, res);
            Ok(false)
        }
        "CODEC_DECODE" => {
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

            match session.decode_codec_frames(tk_ref, &codes, n_frames) {
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
            let synth_res = session.synthesize_speech(tk_ref, &params, Some(&SHUTDOWN_REQ), |samples| {
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
        "EMBED" => {
            if parts.len() < 5 {
                write_reply_res(&mut stream, Err("Malformed EMBED header".into()));
                return Ok(false);
            }
            let strat_str = parts[1];
            let is_query = parts[2] == "1";
            let text_len: usize = parts[3].parse().unwrap_or(0);
            let lora_path = parts[4..].join("\t");
            let mut tbuf = vec![0u8; text_len];
            if text_len > 0 {
                stream.read_exact(&mut tbuf).map_err(|e| e.to_string())?;
            }
            let text = String::from_utf8_lossy(&tbuf);
            let res = EmbedStrategy::from_str(strat_str).and_then(|strat| {
                session.embed_text(
                    strat,
                    &text,
                    is_query,
                    if lora_path.is_empty() { None } else { Some(lora_path.as_str()) },
                )
            });
            match res {
                Ok((vec_f32, ms)) => {
                    let hdr = format!("OK\t{:.3}\t{}\n", ms, vec_f32.len());
                    let mut resp = Vec::with_capacity(hdr.len() + vec_f32.len() * 4);
                    resp.extend_from_slice(hdr.as_bytes());
                    for &v in &vec_f32 {
                        resp.extend_from_slice(&v.to_bits().to_le_bytes());
                    }
                    let _ = stream.write_all(&resp);
                    let _ = stream.flush();
                }
                Err(e) => write_reply_res(&mut stream, Err(e)),
            }
            Ok(false)
        }
        "RAG_STATUS" => {
            write_reply_res(&mut stream, Ok(session.rag_status_json()));
            Ok(false)
        }
        "RAG_CLEAR" => {
            session.rag_clear();
            write_reply_res(&mut stream, Ok("{\"active\":false,\"num_docs\":0}".into()));
            Ok(false)
        }
        "RAG_LOAD" => {
            if parts.len() < 2 {
                write_reply_res(&mut stream, Err("Malformed RAG_LOAD".into()));
                return Ok(false);
            }
            let qvec_path = parts[1..].join("\t");
            let res = session.rag_load_json(Path::new(&qvec_path));
            write_reply_res(&mut stream, res);
            Ok(false)
        }
        "RAG_INDEX" => {
            if parts.len() < 5 {
                write_reply_res(&mut stream, Err("Malformed RAG_INDEX".into()));
                return Ok(false);
            }
            let strat_str = parts[1];
            let load_into_vram = parts[2] == "1";
            let jsonl_path = parts[3];
            let out_qvec_path = parts[4];
            let lora_path = if parts.len() > 5 { parts[5..].join("\t") } else { String::new() };
            let res = EmbedStrategy::from_str(strat_str).and_then(|strat| {
                let docs = load_jsonl_corpus(Path::new(jsonl_path))?;
                let n_docs = docs.len();
                let (tot_ms, dps) = session.rag_index(
                    &docs,
                    strat,
                    &lora_path,
                    Path::new(out_qvec_path),
                    load_into_vram,
                )?;
                Ok(format!(
                    "{{\"status\":\"ok\",\"strategy\":\"{}\",\"num_docs\":{},\"out_path\":\"{}\",\"total_ms\":{:.2},\"passages_per_s\":{:.2}}}",
                    strat.as_str(),
                    n_docs,
                    json_escape(out_qvec_path),
                    tot_ms,
                    dps
                ))
            });
            write_reply_res(&mut stream, res);
            Ok(false)
        }
        "RAG_SEARCH" => {
            if parts.len() < 3 {
                write_reply_res(&mut stream, Err("Malformed RAG_SEARCH".into()));
                return Ok(false);
            }
            let top_k: usize = parts[1].parse().unwrap_or(3);
            let qlen: usize = parts[2].parse().unwrap_or(0);
            let mut qbuf = vec![0u8; qlen];
            if qlen > 0 {
                stream.read_exact(&mut qbuf).map_err(|e| e.to_string())?;
            }
            let query = String::from_utf8_lossy(&qbuf);
            match session.rag_search(&query, top_k) {
                Ok(st) => {
                    let hdr = format!(
                        "OK\t{}\t{:.3}\t{:.3}\t{:.3}\t{}\t{}\n",
                        st.hits.len(),
                        st.embed_ms,
                        st.gpu_search_us,
                        st.total_ms,
                        st.num_docs,
                        st.strategy
                    );
                    let mut resp = Vec::with_capacity(hdr.len() + st.hits.len() * 512);
                    resp.extend_from_slice(hdr.as_bytes());
                    for h in &st.hits {
                        let tb = h.title.as_bytes();
                        let xb = h.text.as_bytes();
                        resp.extend_from_slice(&h.id.to_le_bytes());
                        resp.extend_from_slice(&h.score.to_bits().to_le_bytes());
                        resp.extend_from_slice(&(tb.len() as u32).to_le_bytes());
                        resp.extend_from_slice(tb);
                        resp.extend_from_slice(&(xb.len() as u32).to_le_bytes());
                        resp.extend_from_slice(xb);
                    }
                    let _ = stream.write_all(&resp);
                    let _ = stream.flush();
                }
                Err(e) => write_reply_res(&mut stream, Err(e)),
            }
            Ok(false)
        }
        "DRAFTER_STATUS" => {
            write_reply_res(&mut stream, Ok(session.drafter_status_json()));
            Ok(false)
        }
        "DRAFTER_CLEAR" => {
            session.drafter_clear();
            write_reply_res(&mut stream, Ok("{\"active\":false}".into()));
            Ok(false)
        }
        "DRAFTER_LOAD" => {
            let d_ref = parts.get(1).copied().unwrap_or("qwen3:0.6b");
            let prior_p = parts.get(2).copied().unwrap_or("");
            let res = session.drafter_load_json(d_ref, prior_p);
            write_reply_res(&mut stream, res);
            Ok(false)
        }
        "TOPIC_LOAD" => {
            if parts.len() < 2 {
                write_reply_res(&mut stream, Err("Malformed TOPIC_LOAD".into()));
                return Ok(false);
            }
            let bank_dir = parts[1..].join("\t");
            let res = session.topic_load_json(Path::new(&bank_dir));
            write_reply_res(&mut stream, res);
            Ok(false)
        }
        "TOPIC_STATUS" => {
            write_reply_res(&mut stream, Ok(session.topic_status_json()));
            Ok(false)
        }
        "TOPIC_LIST" => {
            write_reply_res(&mut stream, Ok(session.topic_list_json()));
            Ok(false)
        }
        "TOPIC_CLEAR" => {
            session.topic_clear();
            write_reply_res(&mut stream, Ok("{\"active\":false,\"num_topics\":0}".into()));
            Ok(false)
        }
        "TOPIC_ROUTE" => {
            if parts.len() < 5 {
                write_reply_res(&mut stream, Err("Malformed TOPIC_ROUTE".into()));
                return Ok(false);
            }
            let auto_load = parts[1] == "1";
            let top_k: usize = parts[2].parse().unwrap_or(3);
            let min_score_raw: f32 = parts[3].parse().unwrap_or(-1.0);
            let min_score_opt = if min_score_raw >= 0.0 { Some(min_score_raw) } else { None };
            let qlen: usize = parts[4].parse().unwrap_or(0);
            let mut qbuf = vec![0u8; qlen];
            if qlen > 0 {
                stream.read_exact(&mut qbuf).map_err(|e| e.to_string())?;
            }
            let query = String::from_utf8_lossy(&qbuf);
            let res = session.topic_route(&query, top_k, min_score_opt, auto_load).map(
                |(rt, (used_slots, max_slots, embed_pinned))| {
                    let (t_title, t_slug) = rt
                        .topic
                        .as_ref()
                        .map(|t| (t.title.as_str(), t.slug.as_str()))
                        .unwrap_or(("", ""));
                    let hits_j: Vec<String> = rt
                        .hits
                        .iter()
                        .map(|h| {
                            format!(
                                "{{\"id\":{},\"title\":\"{}\",\"score\":{:.4}}}",
                                h.id,
                                json_escape(&h.title),
                                h.score
                            )
                        })
                        .collect();
                    format!(
                        "{{\"matched\":{},\"topic\":\"{}\",\"slug\":\"{}\",\"score\":{:.6},\"lora_dir\":\"{}\",\"system_prompt\":\"{}\",\"embed_ms\":{:.3},\"gpu_search_us\":{:.2},\"swap_ms\":{:.3},\"cached_topic_slots\":{},\"max_topic_slots\":{},\"embed_lora_pinned\":{},\"hits\":[{}]}}",
                        rt.matched,
                        json_escape(t_title),
                        json_escape(t_slug),
                        rt.score,
                        json_escape(&rt.lora_dir),
                        json_escape(&rt.system_prompt),
                        rt.embed_ms,
                        rt.gpu_search_us,
                        rt.swap_ms,
                        used_slots,
                        max_slots,
                        embed_pinned,
                        hits_j.join(",")
                    )
                },
            );
            write_reply_res(&mut stream, res);
            Ok(false)
        }
        "HOOKS_EXTRACT" | "LOGITS_EVAL" => {
            if parts.len() < 2 {
                write_reply_res(&mut stream, Err("Malformed token-vec command".into()));
                return Ok(false);
            }
            let n_toks: usize = parts[1].parse().unwrap_or(0);
            let mut raw = vec![0u8; n_toks * 4];
            if !raw.is_empty() {
                stream.read_exact(&mut raw).map_err(|e| e.to_string())?;
            }
            let toks: Vec<i32> = raw
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            let res = if parts[0] == "HOOKS_EXTRACT" {
                session.extract_residual_hooks(&toks)
            } else {
                session.eval_next_token_logits(&toks)
            };
            match res {
                Ok(vec_f32) => {
                    let hdr = format!("OK\t{}\n", vec_f32.len());
                    let mut resp = Vec::with_capacity(hdr.len() + vec_f32.len() * 4);
                    resp.extend_from_slice(hdr.as_bytes());
                    for &v in &vec_f32 {
                        resp.extend_from_slice(&v.to_bits().to_le_bytes());
                    }
                    let _ = stream.write_all(&resp);
                    let _ = stream.flush();
                }
                Err(e) => write_reply_res(&mut stream, Err(e)),
            }
            Ok(false)
        }
        "MLP_KV_EXTRACT" => {
            if parts.len() < 5 {
                write_reply_res(&mut stream, Err("Malformed MLP_KV_EXTRACT".into()));
                return Ok(false);
            }
            let n_toks: usize = parts[1].parse().unwrap_or(0);
            let pos_start: usize = parts[2].parse().unwrap_or(0);
            let pos_count: usize = parts[3].parse().unwrap_or(0);
            let layers: Vec<usize> = parts[4]
                .split(',')
                .filter_map(|s| s.trim().parse::<usize>().ok())
                .collect();
            let mut raw = vec![0u8; n_toks * 4];
            if !raw.is_empty() {
                stream.read_exact(&mut raw).map_err(|e| e.to_string())?;
            }
            let toks: Vec<i32> = raw
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            match session.extract_mlp_kv_hooks(&toks, &layers, pos_start, pos_count) {
                Ok((keys_f32, res_f32)) => {
                    let hdr = format!("OK\t{}\t{}\n", keys_f32.len(), res_f32.len());
                    let mut resp = Vec::with_capacity(hdr.len() + (keys_f32.len() + res_f32.len()) * 4);
                    resp.extend_from_slice(hdr.as_bytes());
                    for &v in &keys_f32 {
                        resp.extend_from_slice(&v.to_bits().to_le_bytes());
                    }
                    for &v in &res_f32 {
                        resp.extend_from_slice(&v.to_bits().to_le_bytes());
                    }
                    let _ = stream.write_all(&resp);
                    let _ = stream.flush();
                }
                Err(e) => write_reply_res(&mut stream, Err(e)),
            }
            Ok(false)
        }
        "GEN" => {
            if parts.len() < 6 {
                return Err("Malformed GEN header".into());
            }
            let max_tokens: usize = parts[1].parse().unwrap_or(256);
            let temp: f32 = parts[2].parse().unwrap_or(0.0);
            let spec: bool = parts[3] == "1";
            let reset: bool = parts[4] == "1";
            let mode = parts[5];
            let extra_opt = parts.get(6).copied().filter(|s| !s.is_empty());

            let mut len_b = [0u8; 4];
            stream.read_exact(&mut len_b).map_err(|e| e.to_string())?;
            let payload_len = u32::from_le_bytes(len_b) as usize;
            let mut payload = vec![0u8; payload_len];
            if payload_len > 0 {
                stream.read_exact(&mut payload).map_err(|e| e.to_string())?;
            }

            let mut client_alive = true;
            let gen_res = session.stream_generate(
                mode,
                &payload,
                max_tokens,
                temp,
                spec,
                reset,
                extra_opt,
                Some(&SHUTDOWN_REQ),
                |tok_id, piece| {
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
                Ok(stats_json) => {
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
