//! Zero-Crate 100%-on-RTX-3080 Text Embeddings, Multi-Strategy Pooling (`raw_eos`, `prompt_eol`, `echo`, `svd_lora`),
//! & `.qvec` VRAM `BF16` Cosine Vector Database (`c2w_linear_bf16_kernel`).
//!
//! Complies strictly with `AGENTS.md`:
//! - Rule #1: Shared-ancestor Thinker (`Qwen3-4B`) + Embedding (`Qwen3-Embedding-4B`) with `0` separate models in VRAM.
//! - Rule #2: 100.0% of hidden-state extraction, pooling, whitening, `L2`-norm, and vector search runs on RTX 3080 (`sm_86`).
//! - Rule #3: `0` external Rust crates, `0` runtime Docker containers, `0` `.cu` kernel modifications.

use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

use crate::gguf::BpeTokenizer;
use crate::hw::{put_u32, put_u64, CubinModule, HostMappedBuffer, NvGpu, VramBuffer};
pub use crate::safetensors::f32_to_bf16_bits;

const QVEC_MAGIC: &[u8; 4] = b"QVEC";
const QVEC_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbedStrategy {
    RawEos,
    PromptEol,
    Echo,
    SvdLora,
    Oracle,
}

impl EmbedStrategy {
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "raw" | "raw_eos" | "eos" => Ok(Self::RawEos),
            "prompt_eol" | "prompteol" | "eol" => Ok(Self::PromptEol),
            "echo" | "echo_embeddings" => Ok(Self::Echo),
            "svd_lora" | "svd_lora_r32" | "lora" => Ok(Self::SvdLora),
            "oracle" | "qwen3_embed" => Ok(Self::Oracle),
            other => Err(format!(
                "Unknown embedding strategy '{}' (expected raw_eos, prompt_eol, echo, svd_lora, or oracle)",
                other
            )),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::RawEos => "raw_eos",
            Self::PromptEol => "prompt_eol",
            Self::Echo => "echo",
            Self::SvdLora => "svd_lora",
            Self::Oracle => "oracle",
        }
    }

    /// Zero-shot causal hidden-state methods benefit strongly from subtracting the corpus mean vector
    /// (`anisotropic cone` removal) before `L2` normalization.
    pub fn uses_corpus_centering(&self) -> bool {
        matches!(self, Self::PromptEol | Self::Echo)
    }

    /// Format token IDs `(all_tokens, pool_start_idx)` for either a corpus passage (`is_query = false`)
    /// or a retrieval query (`is_query = true`).
    /// `pool_start_idx` marks the inclusive start index of tokens averaged on GPU (`T - 1` for last-token pooling,
    /// or Copy #2 start for `Echo Embeddings`).
    pub fn encode_input(
        &self,
        tok: &BpeTokenizer,
        eos_id: i32,
        bos_id: Option<i32>,
        text: &str,
        is_query: bool,
        max_tokens_cap: usize,
    ) -> (Vec<i32>, usize) {
        let trimmed = text.trim();
        let endoftext_id = tok
            .find_token_id("<|endoftext|>")
            .unwrap_or(eos_id);

        match self {
            Self::RawEos => {
                let mut ids = Vec::with_capacity(256);
                if let Some(b) = bos_id {
                    ids.push(b);
                }
                let mut body = tok.encode(trimmed);
                let budget = max_tokens_cap.saturating_sub(ids.len() + 1).max(8);
                if body.len() > budget {
                    body.truncate(budget);
                }
                ids.extend_from_slice(&body);
                ids.push(endoftext_id);
                let last = ids.len() - 1;
                (ids, last)
            }
            Self::PromptEol => {
                // Symmetric PromptEOL template aligns query and passage at the exact summary colon (`:`)
                let prefix_str = "This Wikipedia text: \"";
                let suffix_str = "\" means in one essence:";
                let mut pre_ids = Vec::new();
                if let Some(b) = bos_id {
                    pre_ids.push(b);
                }
                pre_ids.extend(tok.encode(prefix_str));
                let suf_ids = tok.encode(suffix_str);
                let mut body = tok.encode(trimmed);
                let budget = max_tokens_cap
                    .saturating_sub(pre_ids.len() + suf_ids.len())
                    .max(8);
                if body.len() > budget {
                    body.truncate(budget);
                }
                let mut ids = Vec::with_capacity(pre_ids.len() + body.len() + suf_ids.len());
                ids.extend_from_slice(&pre_ids);
                ids.extend_from_slice(&body);
                ids.extend_from_slice(&suf_ids);
                let last = ids.len() - 1;
                (ids, last)
            }
            Self::Echo => {
                // Springer et al. Echo Embeddings: Copy #2 attends to 100% of Copy #1 under a standard causal mask!
                let p1_hdr = if is_query {
                    "Search question: "
                } else {
                    "Wikipedia passage: "
                };
                let p2_hdr = if is_query {
                    "\nRepeat search question: "
                } else {
                    "\nRepeat Wikipedia passage: "
                };
                let mut h1 = Vec::new();
                if let Some(b) = bos_id {
                    h1.push(b);
                }
                h1.extend(tok.encode(p1_hdr));
                let h2 = tok.encode(p2_hdr);

                let mut body = tok.encode(trimmed);
                let overhead = h1.len() + h2.len() + 2;
                let half_budget = (max_tokens_cap.saturating_sub(overhead) / 2).max(8);
                if body.len() > half_budget {
                    body.truncate(half_budget);
                }
                if body.is_empty() {
                    body.push(endoftext_id);
                }

                let mut ids = Vec::with_capacity(h1.len() + body.len() + h2.len() + body.len());
                ids.extend_from_slice(&h1);
                ids.extend_from_slice(&body);
                ids.extend_from_slice(&h2);
                let pool_start = ids.len();
                ids.extend_from_slice(&body);
                (ids, pool_start)
            }
            Self::SvdLora | Self::Oracle => {
                // Official `Qwen3-Embedding-4B` instruction-aware query + raw passage + `<|endoftext|>` EOS pooling
                let formatted = if is_query {
                    format!(
                        "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:{}",
                        trimmed
                    )
                } else {
                    trimmed.to_string()
                };
                let mut ids = Vec::with_capacity(256);
                if let Some(b) = bos_id {
                    ids.push(b);
                }
                let mut body = tok.encode(&formatted);
                let budget = max_tokens_cap.saturating_sub(ids.len() + 1).max(8);
                if body.len() > budget {
                    body.truncate(budget);
                }
                ids.extend_from_slice(&body);
                ids.push(endoftext_id);
                let last = ids.len() - 1;
                (ids, last)
            }
        }
    }
}

fn json_esc(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}


#[derive(Clone, Debug)]
pub struct RagDoc {
    pub id: u32,
    pub title: String,
    pub text: String,
}

#[derive(Clone, Debug)]
pub struct RagHit {
    pub id: u32,
    pub title: String,
    pub text: String,
    pub score: f32,
}

#[derive(Clone, Debug)]
pub struct RagSearchStats {
    pub hits: Vec<RagHit>,
    pub embed_ms: f64,
    pub gpu_search_us: f64,
    pub total_ms: f64,
    pub strategy: String,
    pub num_docs: usize,
}

/// Co-resident RTX 3080 `BF16` Vector Database (`5.12 MB` VRAM per `1,000` `2560`-D Wikipedia passages).
#[allow(dead_code)]
pub struct GpuVectorDb {
    path: String,
    strategy: EmbedStrategy,
    lora_path: String,
    dim: usize,
    docs: Vec<RagDoc>,
    has_mean: bool,
    mean_vec: Vec<f32>,
    vectors_bf16_gpu: VramBuffer,
    neg_mean_gpu: Option<VramBuffer>,
    scores_buf: HostMappedBuffer,
}

impl GpuVectorDb {
    #[inline]
    pub fn strategy(&self) -> EmbedStrategy {
        self.strategy
    }

    #[inline]
    pub fn lora_path(&self) -> &str {
        &self.lora_path
    }

    #[inline]
    pub fn num_docs(&self) -> usize {
        self.docs.len()
    }

    #[inline]
    pub fn neg_mean_va(&self, strat: EmbedStrategy) -> Option<u64> {
        if self.strategy == strat {
            self.neg_mean_gpu.as_ref().map(|b| b.va_addr)
        } else {
            None
        }
    }

    pub fn status_json(&self, load_ms_opt: Option<f64>) -> String {
        if let Some(load_ms) = load_ms_opt {
            format!(
                "{{\"active\":true,\"path\":\"{}\",\"strategy\":\"{}\",\"num_docs\":{},\"dim\":{},\"vram_mb\":{:.2},\"load_ms\":{:.2}}}",
                json_esc(&self.path),
                self.strategy.as_str(),
                self.docs.len(),
                self.dim,
                self.vram_mb(),
                load_ms
            )
        } else {
            format!(
                "{{\"active\":true,\"path\":\"{}\",\"strategy\":\"{}\",\"num_docs\":{},\"dim\":{},\"vram_mb\":{:.2},\"lora_path\":\"{}\"}}",
                json_esc(&self.path),
                self.strategy.as_str(),
                self.docs.len(),
                self.dim,
                self.vram_mb(),
                json_esc(&self.lora_path)
            )
        }
    }

    pub fn vram_mb(&self) -> f64 {
        let neg_sz = self.neg_mean_gpu.as_ref().map(|b| b.size).unwrap_or(0);
        ((self.vectors_bf16_gpu.size + neg_sz) as f64) / (1024.0 * 1024.0)
    }

    pub fn save_qvec(
        path: &Path,
        strategy: EmbedStrategy,
        lora_path: &str,
        dim: usize,
        mean_opt: Option<&[f32]>,
        vectors_bf16: &[u16],
        docs: &[RagDoc],
    ) -> Result<(), String> {
        if vectors_bf16.len() != docs.len() * dim {
            return Err(format!(
                "save_qvec mismatch: vectors_bf16.len()={} != docs({}) * dim({})",
                vectors_bf16.len(),
                docs.len(),
                dim
            ));
        }
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
        }

        let strat_b = strategy.as_str().as_bytes();
        let lora_b = lora_path.as_bytes();
        let has_mean = mean_opt.is_some();

        let mut out = Vec::with_capacity(64 + dim * 4 + vectors_bf16.len() * 2 + docs.len() * 512);
        out.extend_from_slice(QVEC_MAGIC);
        out.extend_from_slice(&QVEC_VERSION.to_le_bytes());
        out.extend_from_slice(&(dim as u32).to_le_bytes());
        out.extend_from_slice(&(docs.len() as u32).to_le_bytes());
        out.extend_from_slice(&(u32::from(has_mean)).to_le_bytes());

        out.extend_from_slice(&(strat_b.len() as u32).to_le_bytes());
        out.extend_from_slice(strat_b);
        out.extend_from_slice(&(lora_b.len() as u32).to_le_bytes());
        out.extend_from_slice(lora_b);

        if let Some(mu) = mean_opt {
            for &v in mu {
                out.extend_from_slice(&v.to_bits().to_le_bytes());
            }
        }
        for &bf in vectors_bf16 {
            out.extend_from_slice(&bf.to_le_bytes());
        }
        for d in docs {
            let tb = d.title.as_bytes();
            let xb = d.text.as_bytes();
            out.extend_from_slice(&d.id.to_le_bytes());
            out.extend_from_slice(&(tb.len() as u32).to_le_bytes());
            out.extend_from_slice(tb);
            out.extend_from_slice(&(xb.len() as u32).to_le_bytes());
            out.extend_from_slice(xb);
        }

        fs::write(path, out).map_err(|e| format!("Failed writing {}: {}", path.display(), e))
    }

    pub fn load_qvec(gpu: &mut NvGpu, path: &Path, expected_dim: usize) -> Result<Self, String> {
        let raw = fs::read(path).map_err(|e| format!("Failed reading {}: {}", path.display(), e))?;
        if raw.len() < 28 || &raw[0..4] != QVEC_MAGIC {
            return Err(format!("Invalid .qvec file header: {}", path.display()));
        }
        let mut pos = 4usize;
        let read_u32 = |buf: &[u8], p: &mut usize| -> Result<u32, String> {
            if *p + 4 > buf.len() {
                return Err("Unexpected EOF in .qvec".into());
            }
            let v = u32::from_le_bytes(buf[*p..*p + 4].try_into().unwrap());
            *p += 4;
            Ok(v)
        };

        let ver = read_u32(&raw, &mut pos)?;
        if ver != QVEC_VERSION {
            return Err(format!("Unsupported .qvec version {}", ver));
        }
        let dim = read_u32(&raw, &mut pos)? as usize;
        if expected_dim > 0 && dim != expected_dim {
            return Err(format!(
                ".qvec embedding dim {} does not match loaded Thinker dim {}",
                dim, expected_dim
            ));
        }
        let num_docs = read_u32(&raw, &mut pos)? as usize;
        let has_mean = read_u32(&raw, &mut pos)? != 0;

        let strat_len = read_u32(&raw, &mut pos)? as usize;
        if pos + strat_len > raw.len() {
            return Err("Corrupt strategy string in .qvec".into());
        }
        let strat_str = std::str::from_utf8(&raw[pos..pos + strat_len]).map_err(|e| e.to_string())?;
        pos += strat_len;
        let strategy = EmbedStrategy::from_str(strat_str)?;

        let lora_len = read_u32(&raw, &mut pos)? as usize;
        if pos + lora_len > raw.len() {
            return Err("Corrupt lora_path in .qvec".into());
        }
        let lora_path = String::from_utf8_lossy(&raw[pos..pos + lora_len]).into_owned();
        pos += lora_len;

        let mut mean_vec = vec![0.0f32; dim];
        let neg_mean_gpu = if has_mean {
            if pos + dim * 4 > raw.len() {
                return Err("Corrupt mean_vec in .qvec".into());
            }
            let mut neg_bytes = Vec::with_capacity(dim * 4);
            for i in 0..dim {
                let bits = u32::from_le_bytes(raw[pos..pos + 4].try_into().unwrap());
                pos += 4;
                let v = f32::from_bits(bits);
                mean_vec[i] = v;
                neg_bytes.extend_from_slice(&(-v).to_bits().to_le_bytes());
            }
            let nb = gpu.alloc_vram((neg_bytes.len() as u64).max(4096))?;
            gpu.upload_to_vram(&nb, 0, &neg_bytes)?;
            Some(nb)
        } else {
            None
        };

        let vec_bytes_len = num_docs * dim * 2;
        if pos + vec_bytes_len > raw.len() {
            return Err("Corrupt BF16 vector matrix in .qvec".into());
        }
        let vec_bytes = &raw[pos..pos + vec_bytes_len];
        pos += vec_bytes_len;

        let vectors_bf16_gpu = gpu.alloc_vram((vec_bytes_len as u64).max(4096))?;
        gpu.upload_to_vram(&vectors_bf16_gpu, 0, vec_bytes)?;

        let mut docs = Vec::with_capacity(num_docs);
        for _ in 0..num_docs {
            let id = read_u32(&raw, &mut pos)?;
            let tlen = read_u32(&raw, &mut pos)? as usize;
            if pos + tlen > raw.len() {
                return Err("Corrupt doc title in .qvec".into());
            }
            let title = String::from_utf8_lossy(&raw[pos..pos + tlen]).into_owned();
            pos += tlen;
            let xlen = read_u32(&raw, &mut pos)? as usize;
            if pos + xlen > raw.len() {
                return Err("Corrupt doc text in .qvec".into());
            }
            let text = String::from_utf8_lossy(&raw[pos..pos + xlen]).into_owned();
            pos += xlen;
            docs.push(RagDoc { id, title, text });
        }

        let scores_buf = gpu.alloc_host_mapped((num_docs * 4).max(4096))?;

        Ok(Self {
            path: path.display().to_string(),
            strategy,
            lora_path,
            dim,
            docs,
            has_mean,
            mean_vec,
            vectors_bf16_gpu,
            neg_mean_gpu,
            scores_buf,
        })
    }

    /// Executes 100%-on-RTX-3080 exact inner-product / cosine search (`[1, dim] x [num_docs, dim]^T`)
    /// via `c2w_linear_bf16_kernel` (`float4` + `uint2` `__ldg`), returning top-`k` hits + GPU search time in microseconds!
    pub fn search_cosine_on_gpu(
        &self,
        gpu: &mut NvGpu,
        module: &CubinModule,
        query_f32_gpu_va: u64,
        top_k: usize,
    ) -> Result<(Vec<RagHit>, f64), String> {
        let num_docs = self.docs.len();
        if num_docs == 0 {
            return Ok((Vec::new(), 0.0));
        }

        let t0 = Instant::now();
        let k_lin_bf16 = module.kernel("c2w_linear_bf16_kernel")?;
        let mut args = [0u8; 60];
        put_u64(&mut args, 0, self.scores_buf.va_addr);
        put_u64(&mut args, 8, self.vectors_bf16_gpu.va_addr);
        put_u64(&mut args, 16, query_f32_gpu_va);
        put_u64(&mut args, 24, 0); // bias = null
        put_u64(&mut args, 32, 0); // layer_scale = null
        put_u64(&mut args, 40, 0); // residual = null
        put_u32(&mut args, 48, 1); // T = 1 query
        put_u32(&mut args, 52, num_docs as u32);
        put_u32(&mut args, 56, self.dim as u32);

        gpu.launch_kernel_sync(k_lin_bf16, (((num_docs as u32) + 3) / 4, 1, 1), (128, 1, 1), &args, true, Duration::from_secs(5))?;
        let gpu_us = t0.elapsed().as_secs_f64() * 1e6;

        let scores_ptr = self.scores_buf.as_ptr() as *const f32;
        let mut idx_scores: Vec<(usize, f32)> = (0..num_docs)
            .map(|i| {
                let sc = unsafe { std::ptr::read_volatile(scores_ptr.add(i)) };
                (i, if sc.is_finite() { sc } else { -1e30 })
            })
            .collect();
        let k = top_k.clamp(1, num_docs);
        idx_scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let hits = idx_scores[..k]
            .iter()
            .map(|&(idx, score)| {
                let d = &self.docs[idx];
                RagHit {
                    id: d.id,
                    title: d.title.clone(),
                    text: d.text.clone(),
                    score,
                }
            })
            .collect();

        Ok((hits, gpu_us))
    }
}

// -----------------------------------------------------------------------------
// Zero-Crate JSONL Corpus Parser (`{"id": 0, "title": "...", "text": "..."}`)
// -----------------------------------------------------------------------------

fn extract_json_str_field(line: &str, key_pat: &str) -> Option<String> {
    let idx = line.find(key_pat)?;
    let rest = line[idx + key_pat.len()..].trim_start().strip_prefix(':')?.trim_start().strip_prefix('"')?;
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

fn extract_json_u32_field(line: &str, key_pat: &str) -> Option<u32> {
    let idx = line.find(key_pat)?;
    let rest = line[idx + key_pat.len()..].trim_start().strip_prefix(':')?.trim_start();
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse::<u32>().ok()
}

pub fn load_jsonl_corpus(path: &Path) -> Result<Vec<RagDoc>, String> {
    let content = fs::read_to_string(path)
        .map_err(|e| format!("Failed reading JSONL {}: {}", path.display(), e))?;
    let mut docs = Vec::new();
    for (line_no, raw_ln) in content.lines().enumerate() {
        let ln = raw_ln.trim();
        if ln.is_empty() {
            continue;
        }
        let id = extract_json_u32_field(ln, "\"id\"").unwrap_or(line_no as u32);
        let title = extract_json_str_field(ln, "\"title\"").unwrap_or_else(|| format!("Doc {}", id));
        let text = extract_json_str_field(ln, "\"text\"")
            .or_else(|| extract_json_str_field(ln, "\"context\""))
            .ok_or_else(|| format!("Line {} in {} missing \"text\" field", line_no + 1, path.display()))?;
        docs.push(RagDoc { id, title, text });
    }
    if docs.is_empty() {
        return Err(format!("No documents found in {}", path.display()));
    }
    Ok(docs)
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct RagEvalQuery {
    pub qid: u32,
    pub target_id: u32,
    pub title: String,
    pub question: String,
    pub answer: String,
}

pub fn load_jsonl_eval_queries(path: &Path) -> Result<Vec<RagEvalQuery>, String> {
    let content = fs::read_to_string(path)
        .map_err(|e| format!("Failed reading eval JSONL {}: {}", path.display(), e))?;
    let mut queries = Vec::new();
    for (line_no, raw_ln) in content.lines().enumerate() {
        let ln = raw_ln.trim();
        if ln.is_empty() {
            continue;
        }
        let qid = extract_json_u32_field(ln, "\"qid\"").unwrap_or(line_no as u32);
        let target_id = extract_json_u32_field(ln, "\"target_id\"").ok_or_else(|| {
            format!("Line {} in {} missing \"target_id\"", line_no + 1, path.display())
        })?;
        let title = extract_json_str_field(ln, "\"title\"").unwrap_or_default();
        let question = extract_json_str_field(ln, "\"question\"").ok_or_else(|| {
            format!("Line {} in {} missing \"question\"", line_no + 1, path.display())
        })?;
        let answer = extract_json_str_field(ln, "\"answer\"").unwrap_or_default();
        queries.push(RagEvalQuery {
            qid,
            target_id,
            title,
            question,
            answer,
        });
    }
    if queries.is_empty() {
        return Err(format!("No evaluation queries found in {}", path.display()));
    }
    Ok(queries)
}

pub fn format_rag_augmented_prompt(question: &str, hits: &[RagHit]) -> String {
    if hits.is_empty() {
        return question.to_string();
    }
    let mut ctx = String::from(
        "Use the following retrieved Wikipedia passages to answer the user's question accurately and concisely. Cite the relevant Wikipedia article title(s).\n\n",
    );
    for (i, h) in hits.iter().enumerate() {
        ctx.push_str(&format!(
            "[Wikipedia #{}: {} (similarity {:.3})]\n{}\n\n",
            i + 1,
            h.title,
            h.score,
            h.text
        ));
    }
    ctx.push_str(&format!("User Question: {}", question));
    ctx
}
