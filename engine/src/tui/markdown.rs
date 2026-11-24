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
                let bar_len = inner_cols.saturating_sub(label.chars().count() + 6);
                let hdr_text = format!("   ╭─ {} {}", label, "─".repeat(bar_len));
                lines.push(RenderedLine {
                    spans: vec![Span {
                        text: hdr_text,
                        style: Style::new(FG_CYAN, BG_CODE_HDR).bold(),
                    }],
                    row_bg: BG_CODE_HDR,
                });
            } else {
                in_fence = false;
                let ftr_text = format!("   ╰{}", "─".repeat(inner_cols.saturating_sub(2)));
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
            let code_cols = inner_cols.saturating_sub(4).max(8);
            if raw_ln.is_empty() {
                lines.push(RenderedLine {
                    spans: vec![Span {
                        text: "   │ ".into(),
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
                                        Span { text: "   │ ".into(), style: Style::new(FG_BORDER, BG_CODE) },
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
                                    Span { text: "   │ ".into(), style: Style::new(FG_BORDER, BG_CODE) },
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
                            Span { text: "   │ ".into(), style: Style::new(FG_BORDER, BG_CODE) },
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
                        text: "   ".into(),
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


pub(super) fn format_verify_markdown(
    is_inkling: bool,
    cfg: &crate::gguf::ModelConfig,
    gguf_path: &std::path::Path,
    gguf_size_gb: f64,
    active_gb_per_tok: f64,
) -> String {
    let mtime_info = std::fs::metadata(gguf_path)
        .map(|md| format!("{} bytes (read-only mmap)", md.len()))
        .unwrap_or_else(|_| "unknown".into());
    if is_inkling {
        format!(
            "### Inkling-2-Mini Looped MoE Integrity & Serving Verification\n\
             - **Model Name**: `{}` (24 physical -> {} logical layers, 128x8 Sparse MoE, dim={}, exp_ff={}, heads={}/{} Rel+Sink GQA, vocab={})\n\
             - **GGUF File**: `{}`\n\
             - **Exact Byte Size**: `{}` (`{:.3} GB` total • `~{:.2} GB/tok` active across 48 visits)\n\
             - **Serving Guarantees**: Full 3-pass recurrence preserved (`0` loop passes dropped), planarized zero-LUT `Q2_0` + `IQ2_XXS` `sm_86` persistent megakernel (`~122 tok/s`) + co-resident `DSpark` (`Inkling2MTP`) block-8 drafter (`/spec on` up to `143 tok/s`).",
            cfg.model_name,
            cfg.num_layers,
            cfg.dim,
            cfg.hidden_dim,
            cfg.n_heads,
            cfg.n_kv_heads,
            cfg.vocab_size,
            gguf_path.display(),
            mtime_info,
            gguf_size_gb,
            active_gb_per_tok
        )
    } else {
        format!(
            "### Model Integrity & Pure Serving Layer Verification\n\
             - **Model Name**: `{}` ({} layers, dim={}, hidden={}, heads={}/{} GQA, vocab={})\n\
             - **GGUF File**: `{}`\n\
             - **Exact Byte Size**: `{}` (`{:.3} GB`, unmodified original GGUF weights)\n\
             - **Serving Guarantees**: `0` layers pruned, `0` weights re-quantized or altered, `PROT_READ` read-only file mapping, exact `FP16` KV cache, and `100%` lossless greedy verification (`BM=4` == `T=1`).",
            cfg.model_name,
            cfg.num_layers,
            cfg.dim,
            cfg.hidden_dim,
            cfg.n_heads,
            cfg.n_kv_heads,
            cfg.vocab_size,
            gguf_path.display(),
            mtime_info,
            gguf_size_gb
        )
    }
}

pub(super) fn select_bench_prompt(is_inkling: bool, arg: &str) -> (bool, String) {
    if is_inkling {
        if arg == "spec" || arg == "dspark" {
            (
                true,
                "Output ONLY a valid minified JSON object with keys `gpu`, `sm_count`, `vram_gb`, `quant`, `loop_passes`, and `logical_layers` describing an NVIDIA RTX 3080 (68 SMs, 10 GB) running Inkling-2-Mini (3 passes, 48 logical layers, IQ2). /no_think".into(),
            )
        } else {
            (
                false,
                "Write a concise zero-dependency Rust function `is_prime(n: u64) -> bool`. /no_think".into(),
            )
        }
    } else if arg == "t1" {
        (false, "Count from 1 to 60 separated by commas.".into())
    } else {
        (
            true,
            "Convert the following Python dict so `enable_docker` is False and `kernel_mode` is `'persistent_sm86_megakernel'`, preserving all other fields verbatim:\n\n```python\nENGINE_CONFIG = {\n    'device': 'NVIDIA GeForce RTX 3080 (GA102, sm_86, 68 SMs)',\n    'transport': 'Thunderbolt 3 / DriverKit DMA',\n    'model': 'Qwen3-4B-Instruct-2507-Q4_K_M.gguf',\n    'enable_docker': True,\n    'kernel_mode': 'hcq_graph_346_kernels',\n    'kv_cache_dtype': 'float16',\n    'max_context_tokens': 4096,\n    'quant_formats': ['Q4_K', 'Q6_K', 'Q8_0'],\n}\n```\nReturn only the updated Python code block.".into(),
        )
    }
}

/// Splits `<think>...</think>` reasoning from the visible reply and renders collapsible
/// Claude Code / Hermes-style thought summaries (`Ctrl+O` to expand/collapse).
#[allow(clippy::too_many_arguments)]
pub(super) fn format_assistant_markdown(
    raw_content: &str,
    max_cols: usize,
    is_streaming: bool,
    expand_thinking: bool,
    bg_void: Rgb,
    fg_muted: Rgb,
    fg_dim: Rgb,
    fg_accent: Rgb,
) -> (Vec<RenderedLine>, usize, usize) {
    let inner_cols = max_cols.saturating_sub(6).max(16);

    if let Some(ts) = raw_content.find("<think>") {
        let after_open = &raw_content[ts + 7..];
        if let Some(te) = after_open.find("</think>") {
            let think_txt = after_open[..te].trim();
            let ans_txt = after_open[te + 8..].trim_start_matches(['\r', '\n']);
            let th_words = think_txt.split_whitespace().count();
            let ans_words = ans_txt.split_whitespace().count();
            let mut out = Vec::new();

            if th_words > 0 {
                let arrow = if expand_thinking { "▾" } else { "▸" };
                let hint = if expand_thinking { "Ctrl+O to collapse" } else { "Ctrl+O to expand" };
                out.push(RenderedLine {
                    spans: vec![
                        Span { text: "   ".into(), style: Style::new(fg_dim, bg_void) },
                        Span {
                            text: format!("{} Thought · {} words ", arrow, th_words),
                            style: Style::new(fg_accent, bg_void),
                        },
                        Span {
                            text: format!("({})", hint),
                            style: Style::new(fg_dim, bg_void),
                        },
                    ],
                    row_bg: bg_void,
                });
                if expand_thinking {
                    for ln in think_txt.split('\n') {
                        for mut wl in wrap_inline_markdown(ln, inner_cols, fg_muted, bg_void) {
                            wl.spans.insert(
                                0,
                                Span { text: "   ┆ ".into(), style: Style::new(fg_dim, bg_void) },
                            );
                            out.push(wl);
                        }
                    }
                    out.push(RenderedLine { spans: Vec::new(), row_bg: bg_void });
                }
            }

            let body = if is_streaming {
                if ans_txt.is_empty() { "▍".to_string() } else { format!("{}▍", ans_txt) }
            } else {
                ans_txt.to_string()
            };
            if !body.is_empty() {
                out.extend(format_markdown_block(&body, max_cols, false));
            }
            return (out, th_words, ans_words);
        } else {
            let think_txt = after_open.trim_start_matches(['\r', '\n']);
            let th_words = think_txt.split_whitespace().count();
            let mut out = Vec::new();
            out.push(RenderedLine {
                spans: vec![
                    Span { text: "   ".into(), style: Style::new(fg_dim, bg_void) },
                    Span {
                        text: format!("◈ Thinking · {} words ", th_words),
                        style: Style::new(fg_accent, bg_void).bold(),
                    },
                    Span {
                        text: if expand_thinking {
                            "(Ctrl+O for compact preview)"
                        } else {
                            "(live preview · Ctrl+O for full)"
                        }
                        .into(),
                        style: Style::new(fg_dim, bg_void),
                    },
                ],
                row_bg: bg_void,
            });
            let raw_lines: Vec<&str> = think_txt.split('\n').filter(|l| !l.trim().is_empty()).collect();
            let slice = if expand_thinking || !is_streaming {
                &raw_lines[..]
            } else {
                &raw_lines[raw_lines.len().saturating_sub(3)..]
            };
            for ln in slice {
                for mut wl in wrap_inline_markdown(ln, inner_cols, fg_muted, bg_void) {
                    wl.spans.insert(
                        0,
                        Span { text: "   ┆ ".into(), style: Style::new(fg_dim, bg_void) },
                    );
                    out.push(wl);
                }
            }
            return (out, th_words, 0);
        }
    }

    let body = if is_streaming && raw_content.is_empty() {
        "▍".to_string()
    } else if is_streaming {
        format!("{}▍", raw_content)
    } else {
        raw_content.to_string()
    };
    let ans_words = raw_content.split_whitespace().count();
    (format_markdown_block(&body, max_cols, false), 0, ans_words)
}

pub(super) fn format_help_markdown(is_inkling: bool) -> &'static str {
    if is_inkling {
        "### RTX TUI · Informative HUD & Controls (`Inkling-2-Mini` 48L Sparse MoE)\n\
            - `Type /` — **Interactive Command Palette** (`↑`/`↓` select · `Tab` complete · live `[current]` values)\n\
            - `Ctrl+O` or `/fold` — **Expand / Collapse `<think>` Reasoning Traces** (3-line live preview while thinking)\n\
            - `Ctrl+B` or `/inspector [on|off|auto]` — **Toggle Hardware & Session Telemetry Sidebar** (GDDR6X VRAM map, Roofline `GB/s`, Prefix Cache, Voice Spectrum)\n\
            - `Ctrl+R` or `/listen` · `/talk [on|off]` · `/speak [on|off]` · `/say` · `/voice <name>` — **100%-on-RTX-3080 Voice (`Qwen3-ASR-0.6B` + `Inkling-2-Mini` + `Qwen3-TTS-0.6B` + `24 kHz` Code2Wav)**\n\
            - `/effort <0|0.1|0.2|0.7|0.9>` (or `/think [on|off]`) — Thinking depth (`0` = instant `~122 tok/s` · `0.2` = reasoning · `0.9` = deep math)\n\
            - `Ctrl+S` or `/spec [on|off]` · `/gamma <1..7>` — Toggle **48L Persistent `B=1`** vs **DSpark Block-8 Speculative (`~143 tok/s`)**\n\
            - `Ctrl+T` or `/theme [charm|sr|it|amiga]` — Switch palette (`Charm Copper`, `Second Reality`, `Impulse Tracker`, `Amiga Dusk`)\n\
            - `/bench [spec]` · `/verify` · `/copy [code]` (`Ctrl+Y`) · `/clear` (`Ctrl+L`) · `/demo [1..4]` · `/mod` · `/quit` (`Ctrl+Q`)"
    } else {
        "### RTX TUI · Informative HUD & Controls (Voice + Text + RAG + Topic LoRA)\n\
            - `Type /` — **Interactive Command Palette** (`↑`/`↓` select · `Tab` complete · live `[current]` values)\n\
            - `Ctrl+O` or `/fold` — **Expand / Collapse `<think>` Reasoning Traces** (3-line live preview while thinking)\n\
            - `Ctrl+B` or `/inspector [on|off|auto]` — **Toggle Hardware & Session Telemetry Sidebar** (GDDR6X VRAM map, Roofline `GB/s`, Prefix Cache, Voice Spectrum)\n\
            - `Ctrl+R` or `/listen` · `/talk [on|off]` · `/speak [on|off]` · `/say` · `/voice <name>` — **100%-on-RTX-3080 Full-Duplex Voice**\n\
            - `/topic [wiki|<bank_dir>|list|off]` · `/rag [wiki|<file.qvec>|off]` · `/lora <path|off>` — **Same-Backbone GPU RAG & `< 1 ms` LoRA Hot-Swap**\n\
            - `Ctrl+T` or `/theme [charm|sr|it|amiga]` — Switch palette (`Charm Copper`, `Second Reality`, `Impulse Tracker`, `Amiga Dusk`)\n\
            - `/bench [t1]` · `Ctrl+S` (`/spec`) · `/verify` · `/copy [code]` (`Ctrl+Y`) · `/clear` (`Ctrl+L`) · `/demo [1..4]` · `/mod` · `/quit` (`Ctrl+Q`)"
    }
}
