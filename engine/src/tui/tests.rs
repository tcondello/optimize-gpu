    use super::*;
    use crate::runtime::session::format_inkling_stats_json;
    use crate::models::inkling::{InklingGenStats, TOK_CONTENT_THINK};

    #[test]
    fn test_inkling_stats_json_and_multiturn_tui_prefix_cache() {
        let st_in = InklingGenStats {
            model_name: "Inkling-2-Mini".into(),
            loop_passes: 3,
            logical_layers_per_tok: 48,
            prompt_tokens: 45,
            cached_prefix_tokens: 31,
            completion_tokens: 72,
            prefill_ms: 42.5,
            decode_ms: 598.2,
            total_ms: 640.7,
            prefill_tok_per_s: 329.4,
            decode_tok_per_s: 120.35,
            ttft_ms: 42.5,
            vram_weights_mb: 7522.0,
            vram_kv_conv_mb: 451.66,
            spec_verify_calls: 9,
            spec_accepted_extra: 49,
            gpu_launches: 2,
            stop_reason: "stop",
            token_ids: vec![200004, 200007, 1234, 200001, 200004, 200005, 5678, 200001],
        };
        let j = format_inkling_stats_json(&st_in);
        let parsed = parse_stats_json(&j);
        assert!((parsed.decode_tok_s - 120.35).abs() < 1e-2);
        assert_eq!(parsed.cached_prefix_tokens, 31);
        assert_eq!(parsed.spec_accepted_extra, 49);
        assert_eq!(parsed.kernel_launches, 2);
        assert_eq!(parsed.token_ids, st_in.token_ids);

        let Ok(gguf_path) = resolve_model_path("inkling") else {
            return;
        };
        if !gguf_path.exists() {
            return;
        }
        let gguf_bytes = std::fs::metadata(&gguf_path).map(|m| m.len()).unwrap_or(0);
        let gguf = GgufFile::open(&gguf_path).expect("open inkling gguf");
        let tok = gguf.build_tokenizer().expect("load inkling tokenizer");
        let cfg = ModelConfig::from_gguf(&gguf, &tok, 4096, "inkling").expect("inkling cfg");
        assert_eq!(cfg.arch, "inkling2_loop");
        assert_eq!(cfg.num_layers, 48);
        assert_eq!(cfg.vocab_size, 200058);

        let mut app = TuiApp {
            model_ref: "inkling".into(),
            gguf_path,
            gguf_size_gb: gguf_bytes as f64 / 1e9,
            active_gb_per_tok: 2.68,
            is_inkling: true,
            inkling_effort: "0.2".into(),
            inkling_gamma: 7,
            tok,
            cfg,
            active_sms: 68,
            ping_ms: 0.12,
            daemon_online: true,
            spec_enabled: false,
            temperature: 0.0,
            max_tokens: 512,
            system_prompt: String::new(),
            voice_enabled: false,
            voice_speaker: "ryan".into(),
            voice_lang: "english".into(),
            tts_model_ref: "1.7b".into(),
            tts_speaking: Arc::new(AtomicBool::new(false)),
            tts_cancel_flag: None,
            last_ttfa_ms: Arc::new(std::sync::Mutex::new(0.0)),
            lora_badge: None,
            rag_badge: None,
            rag_top_k: 3,
            topic_badge: None,
            topic_lora_active: false,
            asr_model_ref: "1.7b".into(),
            asr_lang_hint: "auto".into(),
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
            last_tps: 120.35,
            last_ttft_ms: 42.5,
            last_launches: 2,
            last_spec_extra: 0,
            streaming_rx: None,
            cancel_flag: None,
            stream_live_tps: 0.0,
            stream_live_toks: 0,
            spinner_tick: 0,
            toast: None,
            mouse_capture: false,
            demo: DemosceneEngine::new(),
            viz_pcm: Arc::new(Mutex::new(Vec::new())),
            mic_viz_scratch: Vec::new(),
        };

        // Render 1990s demoscene attract banner, all 4 megademo scenes, palettes, + `/verify` + `/help`
        let mut scr = ScreenBuf::new(120, 36);
        let cur0 = app.draw(&mut scr, 120, 36);
        assert!(!scr.render_ansi(cur0).is_empty());
        for scene_idx in 1..=4 {
            let _ = app.handle_slash_command(&format!("/demo {}", scene_idx), None);
            let _ = app.demo.step(Some((&[0.25f32; 256], 24_000)), true, true);
            assert!(app.draw(&mut scr, 120, 36).is_none());
            assert!(scr.render_ansi(None).len() > 1500);
            let _ = app.handle_slash_command("/theme", None);
        }
        let _ = app.handle_slash_command("/demo off", None);
        let _ = app.handle_slash_command("/verify", None);
        let _ = app.handle_slash_command("/help", None);
        let cur1 = app.draw(&mut scr, 120, 36);
        assert!(scr.render_ansi(cur1).len() > 500);
        app.messages.clear();

        // Verify GGUF special-token constants match `BpeTokenizer::encode` 100%
        assert_eq!(
            app.tok.encode("<|message_model|><|content_text|>"),
            vec![TOK_MSG_MODEL, TOK_CONTENT_TEXT]
        );
        assert_eq!(
            app.tok.encode("<|message_model|><|content_thinking|>"),
            vec![TOK_MSG_MODEL, TOK_CONTENT_THINK]
        );
        assert_eq!(app.tok.encode("<|end_message|>"), vec![TOK_END_MESSAGE]);

        // Verify Turn 1 -> Turn 2 bit-exact prefix match on `effort = "0.2"` (thinking tokens)
        app.messages.push(ChatMsg {
            role: "user".into(),
            content: "What is 2 + 2?".into(),
            raw_token_ids: Vec::new(),
            stats: None,
        });
        let p1 = app.encode_conversation(None, None);
        let sim_gen1 = vec![
            TOK_MSG_MODEL,
            TOK_CONTENT_THINK,
            17,
            10,
            17,
            TOK_END_MESSAGE,
            TOK_MSG_MODEL,
            TOK_CONTENT_TEXT,
            19,
            TOK_END_MESSAGE,
        ];
        let mut engine_cached_1 = p1.clone();
        engine_cached_1.extend_from_slice(&sim_gen1);

        app.messages.push(ChatMsg {
            role: "assistant".into(),
            content: "<think>\n2+2\n</think>\n\n4".into(),
            raw_token_ids: sim_gen1,
            stats: Some(parsed),
        });
        app.messages.push(ChatMsg {
            role: "user".into(),
            content: "Multiply that by 10.".into(),
            raw_token_ids: Vec::new(),
            stats: None,
        });
        let p2 = app.encode_conversation(None, None);
        assert!(p2.len() > engine_cached_1.len());
        assert_eq!(&p2[..engine_cached_1.len()], engine_cached_1.as_slice());

        // Also verify `effort = "0"` bit-exact prefix match!
        app.messages.clear();
        app.inkling_effort = "0".into();
        app.messages.push(ChatMsg {
            role: "user".into(),
            content: "Hello".into(),
            raw_token_ids: Vec::new(),
            stats: None,
        });
        let p1_e0 = app.encode_conversation(None, None);
        assert_eq!(
            &p1_e0[p1_e0.len() - 2..],
            &[TOK_MSG_MODEL, TOK_CONTENT_TEXT]
        );
        let sim_gen_e0 = vec![9906, TOK_END_MESSAGE];
        let mut engine_cached_e0 = p1_e0.clone();
        engine_cached_e0.extend_from_slice(&sim_gen_e0);
        app.messages.push(ChatMsg {
            role: "assistant".into(),
            content: "Hi".into(),
            raw_token_ids: sim_gen_e0,
            stats: None,
        });
        app.messages.push(ChatMsg {
            role: "user".into(),
            content: "Next turn".into(),
            raw_token_ids: Vec::new(),
            stats: None,
        });
        let p2_e0 = app.encode_conversation(None, None);
        assert_eq!(&p2_e0[..engine_cached_e0.len()], engine_cached_e0.as_slice());
    }

    #[test]
    fn test_inkling_tui_live_unix_socket_end_to_end() {
        use std::io::{Read, Write};
        use std::os::unix::net::UnixListener;

        let Ok(gguf_path) = resolve_model_path("inkling") else {
            return;
        };
        if !gguf_path.exists() {
            return;
        }

        let sock_path = std::env::temp_dir().join(format!(
            "rtx_test_inkling_{}.sock",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&sock_path);
        let listener = UnixListener::bind(&sock_path).expect("bind temp unix socket");
        std::env::set_var("RTX_DAEMON_SOCK", &sock_path);

        let srv = std::thread::spawn(move || {
            let mut active_gamma = 7usize;
            let mut gen_calls = 0usize;
            let mut turn1_prompt_plus_gen: Vec<i32> = Vec::new();
            while let Ok((mut stream, _)) = listener.accept() {
                let mut hdr = Vec::new();
                let mut b = [0u8; 1];
                while stream.read(&mut b).unwrap_or(0) == 1 {
                    if b[0] == b'\n' {
                        break;
                    }
                    hdr.push(b[0]);
                }
                if hdr.is_empty() {
                    continue;
                }
                let line = String::from_utf8_lossy(&hdr);
                let parts: Vec<&str> = line.split('\t').collect();
                match parts[0] {
                    "PING" => {
                        let _ = stream.write_all(b"PONG\tinkling\tInkling-2-Mini\t68\n");
                    }
                    "INKLING_CFG" => {
                        if let Some(gs) = parts.get(1) {
                            active_gamma = gs.trim().parse().unwrap_or(7);
                        }
                        let _ = stream.write_all(
                            format!("OK\t{{\"gamma\":{},\"dspark_loaded\":true}}\n", active_gamma)
                                .as_bytes(),
                        );
                    }
                    "RESET" => {
                        let _ = stream.write_all(b"OK\n");
                    }
                    "DRAFTER_STATUS" => {
                        let _ = stream.write_all(
                            format!(
                                "OK\t{{\"active\":true,\"model_ref\":\"dspark\",\"model_name\":\"Inkling2MTP-Block8\",\"gamma\":{}}}\n",
                                active_gamma
                            )
                            .as_bytes(),
                        );
                    }
                    "GEN" => {
                        let spec_flag = parts[3] == "1";
                        let mut lb = [0u8; 4];
                        stream.read_exact(&mut lb).unwrap();
                        let plen = u32::from_le_bytes(lb) as usize;
                        let mut payload = vec![0u8; plen];
                        stream.read_exact(&mut payload).unwrap();
                        let prompt_ids: Vec<i32> = payload
                            .chunks_exact(4)
                            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                            .collect();

                        gen_calls += 1;
                        let sim_tokens: &[(i32, &str)] = if gen_calls == 1 {
                            assert!(!spec_flag, "Turn 1 default must be persistent B=1 (spec=false)");
                            &[
                                (TOK_MSG_MODEL, ""),
                                (TOK_CONTENT_THINK, "<think>\n"),
                                (2219, "6*7=42"),
                                (TOK_END_MESSAGE, "\n</think>\n\n"),
                                (TOK_MSG_MODEL, ""),
                                (TOK_CONTENT_TEXT, ""),
                                (2983, "42"),
                                (TOK_END_MESSAGE, ""),
                            ]
                        } else {
                            assert!(spec_flag, "Turn 2 after /spec on must pass spec=true");
                            assert!(
                                prompt_ids.len() > turn1_prompt_plus_gen.len(),
                                "Turn 2 prompt must extend Turn 1 cached sequence"
                            );
                            assert_eq!(
                                &prompt_ids[..turn1_prompt_plus_gen.len()],
                                turn1_prompt_plus_gen.as_slice(),
                                "Turn 2 must match 100% of Turn 1 KV+ShortConv prefix"
                            );
                            &[
                                (TOK_MSG_MODEL, ""),
                                (TOK_CONTENT_TEXT, ""),
                                (5192, "{\"ok\":true}"),
                                (TOK_END_MESSAGE, ""),
                            ]
                        };

                        let mut gen_ids = Vec::new();
                        for &(tid, piece) in sim_tokens {
                            gen_ids.push(tid);
                            let pb = piece.as_bytes();
                            let mut fr = Vec::with_capacity(7 + pb.len());
                            fr.push(0x01);
                            fr.extend_from_slice(&tid.to_le_bytes());
                            fr.extend_from_slice(&(pb.len() as u16).to_le_bytes());
                            fr.extend_from_slice(pb);
                            stream.write_all(&fr).unwrap();
                        }
                        if gen_calls == 1 {
                            turn1_prompt_plus_gen = prompt_ids.clone();
                            turn1_prompt_plus_gen.extend_from_slice(&gen_ids);
                        }
                        let st = InklingGenStats {
                            model_name: "Inkling-2-Mini".into(),
                            loop_passes: 3,
                            logical_layers_per_tok: 48,
                            prompt_tokens: prompt_ids.len(),
                            cached_prefix_tokens: if gen_calls == 1 { 0 } else { turn1_prompt_plus_gen.len() },
                            completion_tokens: gen_ids.len(),
                            prefill_ms: 14.0,
                            prefill_tok_per_s: 1200.0,
                            decode_ms: 58.0,
                            decode_tok_per_s: if spec_flag { 138.00 } else { 120.35 },
                            total_ms: 72.0,
                            ttft_ms: 14.0,
                            gpu_launches: 2,
                            spec_verify_calls: if spec_flag { 1 } else { 0 },
                            spec_accepted_extra: if spec_flag { 3 } else { 0 },
                            stop_reason: "stop",
                            token_ids: gen_ids,
                            vram_weights_mb: 7534.7,
                            vram_kv_conv_mb: 451.7,
                        };
                        let sj = format_inkling_stats_json(&st);
                        let sb = sj.as_bytes();
                        let mut done = Vec::with_capacity(5 + sb.len());
                        done.push(0x02);
                        done.extend_from_slice(&(sb.len() as u32).to_le_bytes());
                        done.extend_from_slice(sb);
                        stream.write_all(&done).unwrap();
                    }
                    "STOP" => {
                        let _ = stream.write_all(b"BYE\n");
                        break;
                    }
                    _ => {
                        let _ = stream.write_all(b"OK\t{}\n");
                    }
                }
            }
            active_gamma
        });

        let mut app = TuiApp::new("inkling", 2048, false, 0.0, 256).expect("TuiApp::new over live socket");
        assert!(app.is_inkling);
        assert!(!app.spec_enabled);
        assert_eq!(app.inkling_gamma, 7);

        // Turn 1: Persistent B=1 with thinking (`effort = "0.2"`)
        app.start_generation("What is 6 * 7?".into(), false);
        while app.streaming_rx.is_some() {
            app.pump_stream();
            std::thread::sleep(Duration::from_millis(2));
        }
        assert!((app.last_tps - 120.35).abs() < 0.05);
        assert!(app.messages.last().unwrap().content.contains("42"));
        assert_eq!(app.messages.last().unwrap().raw_token_ids.len(), 8);

        // Switch to DSpark speculative mode + gamma=5 + Turn 2 prefix-cached follow-up
        let _ = app.handle_slash_command("/spec on", None);
        let _ = app.handle_slash_command("/gamma 5", None);
        assert!(app.spec_enabled);
        assert_eq!(app.inkling_gamma, 5);

        app.start_generation("Now format as JSON.".into(), false);
        while app.streaming_rx.is_some() {
            app.pump_stream();
            std::thread::sleep(Duration::from_millis(2));
        }
        assert!((app.last_tps - 138.00).abs() < 0.05);
        assert!(app.last_cached_prefix > 0);
        assert!(app.messages.last().unwrap().content.contains("{\"ok\":true}"));

        let mut scr = ScreenBuf::new(120, 36);
        let cur = app.draw(&mut scr, 120, 36);
        assert!(scr.render_ansi(cur).len() > 800);

        let _ = DaemonControl::stop();
        let final_gamma = srv.join().expect("join mock daemon");
        assert_eq!(final_gamma, 5);
        let _ = std::fs::remove_file(&sock_path);
        std::env::remove_var("RTX_DAEMON_SOCK");
    }
