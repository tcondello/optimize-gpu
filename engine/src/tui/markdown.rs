//! Terminal Markdown Renderer, Code-Block Extractor, and TTS Speech Sanitizer/Sentence Splitter.

use super::terminal::{
    char_display_width, Rgb, Style, BG_CODE, BG_CODE_HDR, BG_INLINE, BG_MAIN, FG_AMBER, FG_BORDER,
    FG_CODE, FG_CYAN, FG_PURPLE, FG_TEXT,
};

#[derive(Clone)]
pub(super) struct Span {
    pub text: String,
    pub style: Style,
}

#[derive(Clone)]
pub(super) struct RenderedLine {
    pub spans: Vec<Span>,
    pub row_bg: Rgb,
}

fn wrap_inline_markdown(raw_line: &str, max_cols: usize, base_fg: Rgb, row_bg: Rgb) -> Vec<RenderedLine> {
    if max_cols < 8 {
        return vec![RenderedLine { spans: Vec::new(), row_bg }];
    }

    // Parse `**bold**` and `` `code` `` into styled character tokens
    let mut styled_chars: Vec<(char, Style)> = Vec::with_capacity(raw_line.len());
    let chars: Vec<char> = raw_line.chars().collect();
    let mut i = 0usize;
    let mut in_bold = false;
    let mut in_code = false;

    // Heading detection
    let trimmed = raw_line.trim_start();
    let (heading_level, start_idx) = if let Some(rest) = trimmed.strip_prefix("### ") {
        (3usize, raw_line.len() - rest.len())
    } else if let Some(rest) = trimmed.strip_prefix("## ") {
        (2usize, raw_line.len() - rest.len())
    } else if let Some(rest) = trimmed.strip_prefix("# ") {
        (1usize, raw_line.len() - rest.len())
    } else {
        (0usize, 0usize)
    };

    if heading_level > 0 {
        let byte_to_char = raw_line[..start_idx].chars().count();
        i = byte_to_char;
    }

    while i < chars.len() {
        if !in_code && i + 1 < chars.len() && chars[i] == '*' && chars[i + 1] == '*' {
            in_bold = !in_bold;
            i += 2;
            continue;
        }
        if chars[i] == '`' {
            in_code = !in_code;
            i += 1;
            continue;
        }
        let st = if heading_level == 1 {
            Style::new(FG_CYAN, row_bg).bold()
        } else if heading_level == 2 {
            Style::new(FG_AMBER, row_bg).bold()
        } else if heading_level == 3 {
            Style::new(FG_PURPLE, row_bg).bold()
        } else if in_code {
            Style::new(FG_AMBER, BG_INLINE)
        } else if in_bold {
            Style::new(FG_TEXT, row_bg).bold()
        } else {
            Style::new(base_fg, row_bg)
        };
        styled_chars.push((chars[i], st));
        i += 1;
    }

    if styled_chars.is_empty() {
        return vec![RenderedLine { spans: Vec::new(), row_bg }];
    }

    // Word-aware soft wrapping
    let mut out_lines: Vec<RenderedLine> = Vec::new();
    let mut cur_line: Vec<(char, Style)> = Vec::new();
    let mut cur_w = 0usize;

    let push_cur = |line_chars: &[(char, Style)], out: &mut Vec<RenderedLine>| {
        let mut spans: Vec<Span> = Vec::new();
        for &(ch, st) in line_chars {
            if let Some(last) = spans.last_mut() {
                if last.style == st {
                    last.text.push(ch);
                    continue;
                }
            }
            let mut s = String::with_capacity(16);
            s.push(ch);
            spans.push(Span { text: s, style: st });
        }
        out.push(RenderedLine { spans, row_bg });
    };

    let mut pos = 0usize;
    while pos < styled_chars.len() {
        // Group next word or whitespace run
        let is_space = styled_chars[pos].0 == ' ';
        let mut end = pos + 1;
        while end < styled_chars.len() && (styled_chars[end].0 == ' ') == is_space {
            end += 1;
        }
        let word = &styled_chars[pos..end];
        let word_w: usize = word.iter().map(|&(c, _)| char_display_width(c)).sum();

        if cur_w > 0 && cur_w + word_w > max_cols {
            if is_space {
                push_cur(&cur_line, &mut out_lines);
                cur_line.clear();
                cur_w = 0;
                pos = end;
                continue;
            } else if word_w <= max_cols {
                push_cur(&cur_line, &mut out_lines);
                cur_line.clear();
                cur_w = 0;
            }
        }

        for &(c, st) in word {
            let cw = char_display_width(c).max(1);
            if cur_w + cw > max_cols && !cur_line.is_empty() {
                push_cur(&cur_line, &mut out_lines);
                cur_line.clear();
                cur_w = 0;
            }
            cur_line.push((c, st));
            cur_w += cw;
        }
        pos = end;
    }

    if !cur_line.is_empty() || out_lines.is_empty() {
        push_cur(&cur_line, &mut out_lines);
    }
    out_lines
}

pub(super) fn format_markdown_block(content: &str, max_cols: usize, is_user: bool) -> Vec<RenderedLine> {
    let mut lines = Vec::new();
    let inner_cols = max_cols.saturating_sub(4).max(12);
    let mut in_fence = false;

    for raw_ln in content.split('\n') {
        let trimmed = raw_ln.trim_start();
        if let Some(fence_rest) = trimmed.strip_prefix("```") {
            if !in_fence {
                in_fence = true;
                let lang = fence_rest.trim();
                let label = if lang.is_empty() { "code" } else { lang };
                let bar_len = inner_cols.saturating_sub(label.chars().count() + 5);
                let hdr_text = format!("  ╭─ {} {}", label, "─".repeat(bar_len));
                lines.push(RenderedLine {
                    spans: vec![Span {
                        text: hdr_text,
                        style: Style::new(FG_CYAN, BG_CODE_HDR).bold(),
                    }],
                    row_bg: BG_CODE_HDR,
                });
            } else {
                in_fence = false;
                let ftr_text = format!("  ╰{}", "─".repeat(inner_cols.saturating_sub(1)));
                lines.push(RenderedLine {
                    spans: vec![Span {
                        text: ftr_text,
                        style: Style::new(FG_BORDER, BG_CODE),
                    }],
                    row_bg: BG_MAIN,
                });
            }
            continue;
        }

        if in_fence {
            // Hard-wrap code lines cleanly inside code card
            let code_cols = inner_cols.saturating_sub(3).max(8);
            if raw_ln.is_empty() {
                lines.push(RenderedLine {
                    spans: vec![Span {
                        text: "  │ ".into(),
                        style: Style::new(FG_BORDER, BG_CODE),
                    }],
                    row_bg: BG_CODE,
                });
            } else {
                let mut chunk = String::new();
                let mut cw = 0usize;
                for ch in raw_ln.chars() {
                    let ch_exp = if ch == '\t' { "    " } else { "" };
                    if !ch_exp.is_empty() {
                        for sp in ch_exp.chars() {
                            if cw + 1 > code_cols {
                                lines.push(RenderedLine {
                                    spans: vec![
                                        Span { text: "  │ ".into(), style: Style::new(FG_BORDER, BG_CODE) },
                                        Span { text: std::mem::take(&mut chunk), style: Style::new(FG_CODE, BG_CODE) },
                                    ],
                                    row_bg: BG_CODE,
                                });
                                cw = 0;
                            }
                            chunk.push(sp);
                            cw += 1;
                        }
                    } else {
                        let w = char_display_width(ch);
                        if cw + w > code_cols && !chunk.is_empty() {
                            lines.push(RenderedLine {
                                spans: vec![
                                    Span { text: "  │ ".into(), style: Style::new(FG_BORDER, BG_CODE) },
                                    Span { text: std::mem::take(&mut chunk), style: Style::new(FG_CODE, BG_CODE) },
                                ],
                                row_bg: BG_CODE,
                            });
                            cw = 0;
                        }
                        chunk.push(ch);
                        cw += w;
                    }
                }
                if !chunk.is_empty() {
                    lines.push(RenderedLine {
                        spans: vec![
                            Span { text: "  │ ".into(), style: Style::new(FG_BORDER, BG_CODE) },
                            Span { text: chunk, style: Style::new(FG_CODE, BG_CODE) },
                        ],
                        row_bg: BG_CODE,
                    });
                }
            }
        } else {
            let fg = if is_user { Rgb(215, 232, 250) } else { FG_TEXT };
            for mut wl in wrap_inline_markdown(raw_ln, inner_cols, fg, BG_MAIN) {
                wl.spans.insert(
                    0,
                    Span {
                        text: "  ".into(),
                        style: Style::new(FG_TEXT, BG_MAIN),
                    },
                );
                lines.push(wl);
            }
        }
    }
    lines
}

// =============================================================================
// Input Event Decoder (Keyboard + Bracketed Paste + SGR Mouse Wheel)
// =============================================================================


pub(super) fn extract_last_fenced_code(text: &str) -> Option<String> {
    let mut last_block: Option<String> = None;
    let mut cur = String::new();
    let mut inside = false;
    for ln in text.split('\n') {
        if ln.trim_start().starts_with("```") {
            if inside {
                last_block = Some(cur.trim_end_matches('\n').to_string());
                cur.clear();
                inside = false;
            } else {
                inside = true;
                cur.clear();
            }
        } else if inside {
            if !cur.is_empty() {
                cur.push('\n');
            }
            cur.push_str(ln);
        }
    }
    last_block
}

/// Strip fenced code blocks and light Markdown formatting so spoken replies sound natural.
pub(super) fn sanitize_text_for_speech(md: &str) -> String {
    let md = md.rsplit_once("</think>").map(|(_, after)| after).unwrap_or(md);
    let mut out = String::with_capacity(md.len());
    let mut in_fence = false;
    for ln in md.split('\n') {
        let trimmed = ln.trim_start();
        if trimmed.starts_with("```") {
            in_fence = !in_fence;
            if in_fence {
                out.push_str(" Code block omitted. ");
            }
            continue;
        }
        if in_fence {
            continue;
        }
        let cleaned = trimmed
            .trim_start_matches('#')
            .trim_start_matches('-')
            .trim_start();
        for ch in cleaned.chars() {
            if ch != '*' && ch != '`' {
                out.push(ch);
            }
        }
        out.push(' ');
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Split spoken reply text into sentence-sized chunks (~40–180 chars) so the first sentence
/// plays immediately (`< 120 ms` TTFA) while subsequent sentences synthesize seamlessly behind it.
pub(super) fn split_spoken_sentences(clean_text: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut cur = String::new();
    for ch in clean_text.chars() {
        cur.push(ch);
        let len = cur.chars().count();
        let is_sent_end = matches!(ch, '.' | '!' | '?' | ';');
        let is_clause_end = ch == ',' || ch == ':';
        if (is_sent_end && len >= 24) || (chunks.is_empty() && is_clause_end && len >= 42) || len >= 200 && ch == ' ' {
            let trimmed = cur.trim().to_string();
            if !trimmed.is_empty() {
                chunks.push(trimmed);
            }
            cur.clear();
        }
    }
    let tail = cur.trim().to_string();
    if !tail.is_empty() {
        if let Some(last) = chunks.last_mut() {
            if tail.chars().count() < 18 {
                last.push(' ');
                last.push_str(&tail);
                return chunks;
            }
        }
        chunks.push(tail);
    }
    chunks
}


pub(super) fn format_welcome_markdown(
    is_inkling: bool,
    model_name: &str,
    gguf_size_gb: f64,
    active_sms: u32,
    inkling_effort: &str,
) -> String {
    if is_inkling {
        format!(
            "## Welcome to RTX TUI — 100%-on-RTX-3080 Looped Sparse MoE + Voice\n\
             Connected to **{}** (`{:.2} GB` `AD-IQ2_XXS-Q2` • `26.4B` total / `~1.8B` active • `24` physical -> `48` logical layers • `128x8` Sparse MoE) + co-resident **DSpark (`Inkling2MTP`)** on **NVIDIA GeForce RTX 3080 (`GA102`, {} SMs)** with **Docker OFF**.\n\n\
             - **Pareto-Optimal Speed & Reasoning**: Runs the **48-layer Persistent `B=1` `sm_86` Megakernel (`~120 tok/s`)** by default with full 3-pass reasoning (`EFFORT:{}`), plus `< 15 ms` multi-turn KV+ShortConv prefix caching.\n\
             - **Co-Resident RTX 3080 Voice**: Press **`Ctrl+R`** (or `/listen` / `/speak` / `/talk`) for 100%-on-RTX-3080 `Qwen3-ASR-0.6B` STT + `Inkling-2-Mini` + `Qwen3-TTS-0.6B` + `12Hz Code2Wav`!\n\
             - **Controls**: Type `/effort 0` (or `/think off` / append `/no_think`) for instant zero-thinking replies, `/effort 0.2` or `0.9` for reasoning, `/spec on` (`Ctrl+S`) for **DSpark Block-8 Speculative Decoding (up to `143 tok/s` on JSON/code)**, or `/bench` to benchmark live.",
            model_name, gguf_size_gb, active_sms, inkling_effort
        )
    } else {
        format!(
            "## Welcome to RTX TUI — 100%-on-RTX-3080 Text + Voice + RAG + LoRA\n\
             Connected to **{}** (`{:.2} GB` GGUF) + co-resident **Qwen3-ASR-1.7B (`Q8_0`)** & **Qwen3-TTS-1.7B (`Q8_0`)** on **NVIDIA GeForce RTX 3080 (`GA102`, {} SMs)** with **Docker OFF**.\n\n\
             - **Speak to Your RTX 3080**: Press **`Ctrl+R`** (or type `/listen`) and speak a question — 100% of STT, LLM thinking, and `24 kHz` TTS run on the RTX 3080 while capturing the full conversation right here in chat! Or type `/talk` for continuous hands-free conversation.\n\
             - **Pure Serving Layer**: `PROT_READ` read-only GGUF mapping, exact `FP16` KV cache, `< 1 ms` Rank-`r` LoRA hot-swap (`/lora`), same-backbone RAG (`/rag wiki`), & `/bench` for live megakernel verification.",
            model_name, gguf_size_gb, active_sms
        )
    }
}

pub(super) fn format_help_markdown(is_inkling: bool) -> &'static str {
    if is_inkling {
        "### RTX TUI Commands (`Inkling-2-Mini` 48-Layer Looped Sparse MoE)\n\
            - `Ctrl+R` or `/listen` │ `/talk [on|off]` │ `/speak [on|off]` │ `/say` │ `/voice <name>` — **100%-on-RTX-3080 Co-Resident Voice (`Qwen3-ASR-0.6B` + `Inkling-2-Mini` + `Qwen3-TTS-0.6B` + `12Hz Code2Wav`)**\n\
            - `/effort <0|0.1|0.2|0.7|0.9>` (or `/think [on|off]`) — Set thinking effort (`0` = instant `~120 tok/s` • `0.2` = standard reasoning • `0.9` = deep math; or append `/no_think` to any prompt)\n\
            - `/spec [on|off]` (`Ctrl+S`) — Switch between **48L Persistent `B=1` Megakernel (`~120 tok/s` default)** and **DSpark Block-8 Speculative Decoding (up to `143 tok/s` on structured JSON/code)**\n\
            - `/gamma <1..7>` — Tune DSpark block-8 verification depth (`γ=3..7`)\n\
            - `/bench` (or `/bench spec`) — Run live RTX 3080 `~120 tok/s` persistent `B=1` or `~143 tok/s` `DSpark` benchmark\n\
            - `/verify` — Inspect loaded `Inkling-2-Mini` GGUF path, byte size, & 48-layer 128x8 MoE layout\n\
            - `/temp <0.0..2.0>` │ `/max <N>` │ `/sys <text>` — Sampling & system prompt controls\n\
            - `/copy` / `/copy code` (`Ctrl+Y`) — Copy last reply or code block to macOS clipboard\n\
            - `/clear` (`Ctrl+L`) — Clear conversation & KV+ShortConv cache │ `/quit` (`Ctrl+Q`) — Exit │ `/stop` — Unload GPU"
    } else {
        "### RTX TUI Commands (Full-Duplex Voice + Text + RAG + Topic LoRA)\n\
            - `/topic [wiki|<bank_dir>|list|status|off]` — **Sharded Topic Parametric LoRA Bank** (`308 µs` GPU router + `44 µs` cached LoRA swap, `0` context stuffing!)\n\
            - `/rag [wiki|<file.qvec>|off]` — **100%-on-RTX-3080 Same-Model Wikipedia VectorDB RAG** (`< 10 ms` embed + `< 50 µs` GPU cosine search; works with both voice & text!)\n\
            - `Ctrl+R` or `/listen` (`/mic`) — **Speak a question into your Mac mic**: runs 100%-on-RTX-3080 `Qwen3-ASR-1.7B Q8_0` STT -> (optional Wiki RAG) -> streams LLM -> speaks reply via `Qwen3-TTS-1.7B Q8_0`!\n\
            - `/talk [on|off]` — **Hands-free continuous voice conversation** (auto-listens after each spoken reply; `Esc` stops)\n\
            - `/speak [on|off]` — Toggle live **100%-on-RTX-3080 `Qwen3-TTS-1.7B Q8_0`** spoken replies (`24 kHz` streaming)\n\
            - `/say [text]` — Speak `<text>` (or the last assistant reply) aloud on RTX 3080 (`Esc` stops audio)\n\
            - `/voice <name>` — Select voice (`ryan`, `serena`, `vivian`, `aiden`, `uncle_fu`, `ono_anna`, `sohee`, `eric`, `dylan`)\n\
            - `/lora <path|off>` — `< 1 ms` hot-swap Rank-`r` (`r <= 32`) PEFT `.safetensors` LoRA adapter\n\
            - `/bench` (or `/bench t1`) — Run live `BM=4` or `T=1` megakernel benchmark inside chat\n\
            - `/verify` — Inspect loaded GGUF path, exact byte size, & unmodified model architecture\n\
            - `/spec [on|off]` (`Ctrl+S`) — Toggle lossless `BM=4` speculative verification vs pure `T=1`\n\
            - `/temp <0.0..2.0>` │ `/max <N>` │ `/sys <text>` — Sampling & system prompt controls\n\
            - `/copy` / `/copy code` (`Ctrl+Y`) — Copy last reply or code block to macOS clipboard\n\
            - `/clear` (`Ctrl+L`) — Clear conversation & KV cache │ `/quit` (`Ctrl+Q`) — Exit │ `/stop` — Unload GPU"
    }
}
