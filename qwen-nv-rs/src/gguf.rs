//! Zero-Dependency `mmap` GGUF v2/v3 Reader, Model Cache Resolver, RoPE Precomputation,
//! and Raw Quantized (`Q4_K` / `Q6_K` / `Q8_0`) Tensor Extraction.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::hw::{get_u16, get_u32, get_u64};
pub use crate::safetensors::{f16_to_f32, f32_to_bf16_bits, f32_to_f16_bits, MmapFile};
pub use crate::tokenizer::{BpeTokenizer, Utf8StreamDecoder};

// -----------------------------------------------------------------------------
// Pure-Rust RFC 1321 MD5 (for zero-download reuse of `~/Library/Caches/tinygrad/downloads/<md5>`)
// -----------------------------------------------------------------------------

pub fn md5_hex(input: &[u8]) -> String {
    const K: [u32; 64] = [
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
        0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
        0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
        0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
        0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
        0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
        0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
    ];
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
        5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
        4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
        6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];

    let mut msg = input.to_vec();
    let bit_len = (input.len() as u64) * 8;
    msg.push(0x80);
    while (msg.len() % 64) != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_le_bytes());

    let (mut a0, mut b0, mut c0, mut d0) = (0x67452301u32, 0xefcdab89u32, 0x98badcfeu32, 0x10325476u32);
    for chunk in msg.chunks_exact(64) {
        let mut m = [0u32; 16];
        for i in 0..16 {
            m[i] = get_u32(chunk, i * 4);
        }
        let (mut a, mut b, mut c, mut d) = (a0, b0, c0, d0);
        for i in 0..64 {
            let (f, g) = match i {
                0..=15 => ((b & c) | (!b & d), i),
                16..=31 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                32..=47 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let tmp = d;
            d = c;
            c = b;
            b = b.wrapping_add(a.wrapping_add(f).wrapping_add(K[i]).wrapping_add(m[g]).rotate_left(S[i]));
            a = tmp;
        }
        a0 = a0.wrapping_add(a);
        b0 = b0.wrapping_add(b);
        c0 = c0.wrapping_add(c);
        d0 = d0.wrapping_add(d);
    }
    let mut out = [0u8; 16];
    out[0..4].copy_from_slice(&a0.to_le_bytes());
    out[4..8].copy_from_slice(&b0.to_le_bytes());
    out[8..12].copy_from_slice(&c0.to_le_bytes());
    out[12..16].copy_from_slice(&d0.to_le_bytes());
    out.iter().map(|b| format!("{:02x}", b)).collect()
}

pub fn resolve_model_path(model_ref: &str) -> Result<PathBuf, String> {
    if matches!(model_ref, "qwen3-8b-patched" | "qwen8b-patched" | "default") {
        let rel = PathBuf::from("checkpoints/qwen3-8b-patched/Qwen3-8B-Patched-Q6_K.gguf");
        if rel.exists() {
            return Ok(rel);
        }
        let p = PathBuf::from("/Users/tim/Code/optimize-gpu/checkpoints/qwen3-8b-patched/Qwen3-8B-Patched-Q6_K.gguf");
        if p.exists() {
            return Ok(p);
        }
        eprintln!("[qwen-nv] Note: patched checkpoint not found locally; falling back to qwen3-thinking:8b");
        return resolve_model_path("qwen3-thinking:8b");
    }
    if matches!(model_ref, "qwen3-patched" | "qwen-patched" | "qwen3-4b-patched") {
        let rel = PathBuf::from("checkpoints/qwen-patched/Qwen3-4B-Thinking-2507-Patched-Q8_0.gguf");
        if rel.exists() {
            return Ok(rel);
        }
        let p = PathBuf::from("/Users/tim/Code/optimize-gpu/checkpoints/qwen-patched/Qwen3-4B-Thinking-2507-Patched-Q8_0.gguf");
        if p.exists() {
            return Ok(p);
        }
    }
    let url_opt = match model_ref {
        "inkling" | "inkling-2" | "inkling:mini" | "inkling-2-mini" | "inkling2" | "inkling2:mini" => {
            Some("https://huggingface.co/AtomicChat/Inkling-2-Mini-GGUF/resolve/main/Inkling-2-Mini-AD-IQ2_XXS-Q2.gguf")
        }
        "qwen3-thinking:8b" | "qwen3-think:8b" | "qwen3:8b" | "qwen3:8b-q6" => {
            Some("https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q6_K.gguf")
        }
        "qwen3:8b-q8" => Some("https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q8_0.gguf"),
        "qwen3:8b-q4" => Some("https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf"),
        "qwen3-embed:8b" | "qwen3-embedding:8b" | "qwen3-embed:8b-q6" => {
            Some("https://huggingface.co/Qwen/Qwen3-Embedding-8B-GGUF/resolve/main/Qwen3-Embedding-8B-Q6_K.gguf")
        }
        "qwen3-embed:8b-q8" => {
            Some("https://huggingface.co/Qwen/Qwen3-Embedding-8B-GGUF/resolve/main/Qwen3-Embedding-8B-Q8_0.gguf")
        }
        "qwen3:4b" => Some("https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"),
        "qwen3-thinking:4b" | "qwen3-think:4b" => Some("https://huggingface.co/unsloth/Qwen3-4B-Thinking-2507-GGUF/resolve/main/Qwen3-4B-Thinking-2507-Q8_0.gguf"),
        "qwen3-embed:4b" | "qwen3-embedding:4b" => Some("https://huggingface.co/Qwen/Qwen3-Embedding-4B-GGUF/resolve/main/Qwen3-Embedding-4B-Q8_0.gguf"),
        "qwen3:0.6b" => Some("https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf"),
        "qwen3:1.7b" => Some("https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf"),
        "qwen3:14b" => Some("https://huggingface.co/Qwen/Qwen3-14B-GGUF/resolve/main/Qwen3-14B-Q4_K_M.gguf"),
        "qwen2.5:3b" => Some("https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf"),
        "qwen2.5:7b" => Some("https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf"),
        s if s.starts_with("http://") || s.starts_with("https://") => Some(s),
        _ => None,
    };

    if let Some(url) = url_opt {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
        let hash = md5_hex(url.as_bytes());
        let tg_mac = PathBuf::from(&home).join("Library/Caches/tinygrad/downloads").join(&hash);
        if tg_mac.exists() {
            return Ok(tg_mac);
        }
        let tg_xdg = PathBuf::from(&home).join(".cache/tinygrad/downloads").join(&hash);
        if tg_xdg.exists() {
            return Ok(tg_xdg);
        }
        let fname = url.rsplit('/').next().unwrap_or(&hash);
        let local_cache = PathBuf::from(&home).join(".cache/qwen-nv-rs/models").join(fname);
        if local_cache.exists() {
            return Ok(local_cache);
        }
        if let Some(parent) = local_cache.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        eprintln!("[qwen-nv] Downloading {} -> {}...", url, local_cache.display());
        let status = Command::new("curl")
            .args(["-L", "--fail", "-o"])
            .arg(&local_cache)
            .arg(url)
            .status()
            .map_err(|e| format!("Failed to launch curl: {}", e))?;
        if !status.success() {
            return Err(format!("curl failed downloading {}", url));
        }
        return Ok(local_cache);
    }

    let p = PathBuf::from(model_ref);
    if p.is_dir() {
        if let Ok(entries) = std::fs::read_dir(&p) {
            for ent in entries.flatten() {
                let ep = ent.path();
                if ep.extension().and_then(|e| e.to_str()) == Some("gguf") {
                    return Ok(ep);
                }
            }
        }
    }
    if p.exists() {
        Ok(p)
    } else {
        Err(format!("Model path not found: {}", model_ref))
    }
}

pub fn resolve_tts_model_paths(talker_ref: &str, codec_ref: Option<&str>) -> Result<(PathBuf, PathBuf), String> {
    let talker_url_or_path = match talker_ref {
        "" | "1.7b" | "1.7b-q8" | "q8_0" | "default" => {
            "https://huggingface.co/Serveurperso/Qwen3-TTS-GGUF/resolve/main/qwen-talker-1.7b-customvoice-Q8_0.gguf"
        }
        "1.7b-q4" | "q4_k_m" => {
            "https://huggingface.co/Serveurperso/Qwen3-TTS-GGUF/resolve/main/qwen-talker-1.7b-customvoice-Q4_K_M.gguf"
        }
        "1.7b-voicedesign" | "voicedesign" => {
            "https://huggingface.co/Serveurperso/Qwen3-TTS-GGUF/resolve/main/qwen-talker-1.7b-voicedesign-Q8_0.gguf"
        }
        "0.6b" | "0.6b-q8" => {
            "https://huggingface.co/Serveurperso/Qwen3-TTS-GGUF/resolve/main/qwen-talker-0.6b-customvoice-Q8_0.gguf"
        }
        other => other,
    };
    let codec_url_or_path = match codec_ref.unwrap_or("default") {
        "" | "default" | "12hz" | "bf16" => {
            "https://huggingface.co/Serveurperso/Qwen3-TTS-GGUF/resolve/main/qwen-tokenizer-12hz-BF16.gguf"
        }
        other => other,
    };
    let talker_path = resolve_model_path(talker_url_or_path)?;
    let codec_path = resolve_model_path(codec_url_or_path)?;
    Ok((talker_path, codec_path))
}

pub fn resolve_asr_model_path(asr_ref: &str) -> Result<PathBuf, String> {
    let url_or_path = match asr_ref {
        "" | "1.7b" | "1.7b-q8" | "q8_0" | "default" | "qwen3-asr:1.7b" => {
            "https://huggingface.co/handy-computer/Qwen3-ASR-1.7B-gguf/resolve/main/Qwen3-ASR-1.7B-Q8_0.gguf"
        }
        "1.7b-q4" | "q4_k_m" => {
            "https://huggingface.co/handy-computer/Qwen3-ASR-1.7B-gguf/resolve/main/Qwen3-ASR-1.7B-Q4_K_M.gguf"
        }
        "1.7b-q6" | "q6_k" => {
            "https://huggingface.co/handy-computer/Qwen3-ASR-1.7B-gguf/resolve/main/Qwen3-ASR-1.7B-Q6_K.gguf"
        }
        "0.6b" | "0.6b-q8" | "qwen3-asr:0.6b" => {
            "https://huggingface.co/handy-computer/Qwen3-ASR-0.6B-gguf/resolve/main/Qwen3-ASR-0.6B-Q8_0.gguf"
        }
        "0.6b-q4" => {
            "https://huggingface.co/handy-computer/Qwen3-ASR-0.6B-gguf/resolve/main/Qwen3-ASR-0.6B-Q4_K_M.gguf"
        }
        other => other,
    };
    resolve_model_path(url_or_path)
}

// -----------------------------------------------------------------------------
// GGUF v2/v3 Metadata & Tensor Info Parser
// -----------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum GgufVal {
    UInt(u64),
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    StrArr(Vec<String>),
    IntArr(Vec<i32>),
    OtherArr,
}

impl GgufVal {
    fn as_u64(&self) -> Option<u64> {
        match self {
            GgufVal::UInt(v) => Some(*v),
            GgufVal::Int(v) => Some(*v as u64),
            _ => None,
        }
    }
    fn as_f32(&self) -> Option<f32> {
        match self {
            GgufVal::Float(v) => Some(*v as f32),
            GgufVal::UInt(v) => Some(*v as f32),
            GgufVal::Int(v) => Some(*v as f32),
            _ => None,
        }
    }
    fn as_bool(&self) -> Option<bool> {
        match self {
            GgufVal::Bool(b) => Some(*b),
            GgufVal::UInt(v) => Some(*v != 0),
            _ => None,
        }
    }
    fn as_str(&self) -> Option<&str> {
        match self {
            GgufVal::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub dims: Vec<usize>,
    pub ggml_type: i32,
    pub offset: usize,
}

pub struct GgufFile {
    mmap: MmapFile,
    kv: HashMap<String, GgufVal>,
    tensors: HashMap<String, TensorInfo>,
    data_start: usize,
}

impl GgufFile {
    pub fn open(path: &Path) -> Result<Self, String> {
        let mmap = MmapFile::open(path)?;
        let b = mmap.as_slice();
        if b.len() < 24 || &b[0..4] != b"GGUF" {
            return Err(format!("Invalid GGUF header in {}", path.display()));
        }
        let ver = get_u32(b, 4);
        if ver != 2 && ver != 3 {
            return Err(format!("Unsupported GGUF version {}", ver));
        }
        let n_tensors = get_u64(b, 8) as usize;
        let n_kv = get_u64(b, 16) as usize;
        let mut cur = 24usize;

        let read_str = |buf: &[u8], c: &mut usize| -> String {
            let len = get_u64(buf, *c) as usize;
            *c += 8;
            let s = String::from_utf8_lossy(&buf[*c..*c + len]).into_owned();
            *c += len;
            s
        };

        fn skip_scalar(t: u32, c: &mut usize, buf: &[u8]) {
            match t {
                0 | 1 | 7 => *c += 1,
                2 | 3 => *c += 2,
                4 | 5 | 6 => *c += 4,
                10 | 11 | 12 => *c += 8,
                8 => {
                    let l = get_u64(buf, *c) as usize;
                    *c += 8 + l;
                }
                9 => {
                    let et = get_u32(buf, *c);
                    let n = get_u64(buf, *c + 4) as usize;
                    *c += 12;
                    for _ in 0..n {
                        skip_scalar(et, c, buf);
                    }
                }
                _ => {}
            }
        }

        let read_val = |buf: &[u8], c: &mut usize, key: &str| -> GgufVal {
            let t = get_u32(buf, *c);
            *c += 4;
            match t {
                0 => { let v = buf[*c] as u64; *c += 1; GgufVal::UInt(v) }
                1 => { let v = (buf[*c] as i8) as i64; *c += 1; GgufVal::Int(v) }
                2 => { let v = get_u16(buf, *c) as u64; *c += 2; GgufVal::UInt(v) }
                3 => { let v = (get_u16(buf, *c) as i16) as i64; *c += 2; GgufVal::Int(v) }
                4 => { let v = get_u32(buf, *c) as u64; *c += 4; GgufVal::UInt(v) }
                5 => { let v = (get_u32(buf, *c) as i32) as i64; *c += 4; GgufVal::Int(v) }
                6 => {
                    let f = f32::from_bits(get_u32(buf, *c)) as f64;
                    *c += 4;
                    GgufVal::Float(f)
                }
                7 => { let v = buf[*c] != 0; *c += 1; GgufVal::Bool(v) }
                8 => GgufVal::Str(read_str(buf, c)),
                10 => { let v = get_u64(buf, *c); *c += 8; GgufVal::UInt(v) }
                11 => { let v = get_u64(buf, *c) as i64; *c += 8; GgufVal::Int(v) }
                12 => {
                    let f = f64::from_bits(get_u64(buf, *c));
                    *c += 8;
                    GgufVal::Float(f)
                }
                9 => {
                    let et = get_u32(buf, *c);
                    let n = get_u64(buf, *c + 4) as usize;
                    *c += 12;
                    if et == 8 && (key == "tokenizer.ggml.tokens" || n <= 64) {
                        let mut v = Vec::with_capacity(n);
                        for _ in 0..n {
                            v.push(read_str(buf, c));
                        }
                        GgufVal::StrArr(v)
                    } else if (et == 4 || et == 5) && (key == "tokenizer.ggml.token_type" || n <= 64) {
                        let mut v = Vec::with_capacity(n);
                        for _ in 0..n {
                            v.push(get_u32(buf, *c) as i32);
                            *c += 4;
                        }
                        GgufVal::IntArr(v)
                    } else if et == 7 && n <= 64 {
                        let mut v = Vec::with_capacity(n);
                        for _ in 0..n {
                            v.push(if buf[*c] != 0 { 1 } else { 0 });
                            *c += 1;
                        }
                        GgufVal::IntArr(v)
                    } else {
                        for _ in 0..n {
                            skip_scalar(et, c, buf);
                        }
                        GgufVal::OtherArr
                    }
                }
                other => panic!("Unknown GGUF value type {}", other),
            }
        };

        let mut kv = HashMap::with_capacity(n_kv);
        for _ in 0..n_kv {
            let k = read_str(b, &mut cur);
            let v = read_val(b, &mut cur, &k);
            kv.insert(k, v);
        }

        let mut tensors = HashMap::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = read_str(b, &mut cur);
            let ndim = get_u32(b, cur) as usize;
            cur += 4;
            let mut dims = Vec::with_capacity(ndim);
            for _ in 0..ndim {
                dims.push(get_u64(b, cur) as usize);
                cur += 8;
            }
            let ggml_type = get_u32(b, cur) as i32;
            cur += 4;
            let offset = get_u64(b, cur) as usize;
            cur += 8;
            tensors.insert(
                name,
                TensorInfo {
                    dims,
                    ggml_type,
                    offset,
                },
            );
        }

        let alignment = kv
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(32) as usize;
        let data_start = (cur + alignment - 1) & !(alignment - 1);

        Ok(Self {
            mmap,
            kv,
            tensors,
            data_start,
        })
    }

    #[inline]
    pub fn file_len(&self) -> usize {
        self.mmap.len()
    }

    #[inline]
    pub fn data_start(&self) -> usize {
        self.data_start
    }

    #[inline]
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.kv.get(key).and_then(|v| v.as_str())
    }

    #[inline]
    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.kv.get(key).and_then(|v| v.as_u64())
    }

    #[inline]
    pub fn get_usize_or(&self, key: &str, def: usize) -> usize {
        self.get_u64(key).map(|v| v as usize).unwrap_or(def)
    }

    pub fn req_usize(&self, key: &str) -> Result<usize, String> {
        self.get_u64(key)
            .map(|v| v as usize)
            .ok_or_else(|| format!("Missing GGUF key {}", key))
    }

    #[inline]
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.kv.get(key).and_then(|v| v.as_f32())
    }

    #[inline]
    pub fn get_f32_or(&self, key: &str, def: f32) -> f32 {
        self.get_f32(key).unwrap_or(def)
    }

    pub fn req_f32(&self, key: &str) -> Result<f32, String> {
        self.get_f32(key)
            .ok_or_else(|| format!("Missing GGUF key {}", key))
    }

    #[inline]
    pub fn get_bool_or(&self, key: &str, def: bool) -> bool {
        self.kv.get(key).and_then(|v| v.as_bool()).unwrap_or(def)
    }

    pub fn get_str_arr(&self, key: &str) -> Option<&[String]> {
        match self.kv.get(key) {
            Some(GgufVal::StrArr(v)) => Some(v.as_slice()),
            _ => None,
        }
    }

    pub fn get_i32_arr(&self, key: &str) -> Option<&[i32]> {
        match self.kv.get(key) {
            Some(GgufVal::IntArr(v)) => Some(v.as_slice()),
            _ => None,
        }
    }

    #[inline]
    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    #[inline]
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    pub fn req_tensor_info(&self, name: &str) -> Result<&TensorInfo, String> {
        self.tensors
            .get(name)
            .ok_or_else(|| format!("Missing required tensor '{}' in GGUF", name))
    }

    pub fn sorted_tensor_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.tensors.keys().cloned().collect();
        names.sort();
        names
    }

    pub fn tensor_names_by_offset(&self) -> Vec<String> {
        let mut names: Vec<String> = self.tensors.keys().cloned().collect();
        names.sort_by_key(|n| self.tensors[n].offset);
        names
    }

    pub fn total_param_count(&self) -> u64 {
        self.tensors
            .values()
            .map(|t| t.dims.iter().product::<usize>() as u64)
            .sum()
    }

    #[inline]
    pub fn tensor_bytes(&self, info: &TensorInfo, nbytes: usize) -> &[u8] {
        let start = self.data_start + info.offset;
        &self.mmap.as_slice()[start..start + nbytes]
    }

    pub fn build_tokenizer(&self) -> Result<BpeTokenizer, String> {
        let tokens = self
            .get_str_arr("tokenizer.ggml.tokens")
            .ok_or_else(|| "Missing tokenizer.ggml.tokens in GGUF".to_string())?;
        let types_opt = self.get_i32_arr("tokenizer.ggml.token_type");
        let raw_pre = self.get_str("tokenizer.ggml.pre").unwrap_or("qwen2");
        BpeTokenizer::from_vocab(tokens, types_opt, raw_pre)
    }
}

// -----------------------------------------------------------------------------
// Tensor Extraction & Fallback Q8_0 Conversion Helpers
// -----------------------------------------------------------------------------

pub fn decode_1d_f32(gguf: &GgufFile, name: &str) -> Result<Option<Vec<f32>>, String> {
    let Some(info) = gguf.tensor_info(name) else {
        return Ok(None);
    };
    let nel: usize = info.dims.iter().product();
    let mut out = Vec::with_capacity(nel);
    match info.ggml_type {
        0 => {
            let raw = gguf.tensor_bytes(info, nel * 4);
            for i in 0..nel {
                out.push(f32::from_bits(get_u32(raw, i * 4)));
            }
        }
        1 => {
            let raw = gguf.tensor_bytes(info, nel * 2);
            for i in 0..nel {
                out.push(f16_to_f32(get_u16(raw, i * 2)));
            }
        }
        30 => {
            let raw = gguf.tensor_bytes(info, nel * 2);
            for i in 0..nel {
                out.push(f32::from_bits((get_u16(raw, i * 2) as u32) << 16));
            }
        }
        other => return Err(format!("Unsupported 1D norm/bias ggml_type={} on {}", other, name)),
    }
    Ok(Some(out))
}

pub fn decode_tensor_f32_req(gguf: &GgufFile, name: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
    let info = gguf.req_tensor_info(name)?;
    let dims = info.dims.clone();
    let v = decode_1d_f32(gguf, name)?.ok_or_else(|| format!("Missing '{}'", name))?;
    Ok((v, dims))
}

pub fn decode_tensor_bf16_req(gguf: &GgufFile, name: &str) -> Result<(Vec<u16>, Vec<usize>), String> {
    let info = gguf.req_tensor_info(name)?;
    let nel: usize = info.dims.iter().product();
    let mut out = Vec::with_capacity(nel);
    match info.ggml_type {
        30 => {
            let raw = gguf.tensor_bytes(info, nel * 2);
            for i in 0..nel {
                out.push(get_u16(raw, i * 2));
            }
        }
        1 => {
            let raw = gguf.tensor_bytes(info, nel * 2);
            for i in 0..nel {
                out.push(f32_to_bf16_bits(f16_to_f32(get_u16(raw, i * 2))));
            }
        }
        0 => {
            let raw = gguf.tensor_bytes(info, nel * 4);
            for i in 0..nel {
                out.push(f32_to_bf16_bits(f32::from_bits(get_u32(raw, i * 4))));
            }
        }
        other => return Err(format!("Unsupported ggml_type={} for BF16 load on {}", other, name)),
    }
    Ok((out, info.dims.clone()))
}

pub fn raw_quant_byte_len(nel: usize, ggml_type: i32) -> Option<usize> {
    match ggml_type {
        8 => Some((nel / 32) * 34),    // Q8_0
        12 => Some((nel / 256) * 144), // Q4_K
        14 => Some((nel / 256) * 210), // Q6_K
        _ => None,
    }
}

/// Extract raw `Q4_K`/`Q6_K`/`Q8_0` bytes (or convert F16/F32/BF16 linear weights to `Q8_0` once).
pub fn extract_quant_weight_bytes<'a>(
    gguf: &'a GgufFile,
    name: &str,
) -> Result<(std::borrow::Cow<'a, [u8]>, i32, usize, usize), String> {
    let info = gguf.req_tensor_info(name)?;
    let in_f = info.dims[0];
    let out_f = if info.dims.len() > 1 { info.dims[1] } else { 1 };
    let nel = in_f * out_f;

    if let Some(nbytes) = raw_quant_byte_len(nel, info.ggml_type) {
        return Ok((std::borrow::Cow::Borrowed(gguf.tensor_bytes(info, nbytes)), info.ggml_type, out_f, in_f));
    }

    if in_f % 32 != 0 {
        return Err(format!("Fallback tensor {} in_features={} not divisible by 32", name, in_f));
    }
    let n_blks = nel / 32;
    let mut q8 = vec![0u8; n_blks * 34];
    let elem_sz = if info.ggml_type == 0 { 4 } else { 2 };
    let raw = gguf.tensor_bytes(info, nel * elem_sz);

    for b in 0..n_blks {
        let mut vals = [0.0f32; 32];
        let mut amax = 0.0f32;
        for j in 0..32 {
            let idx = b * 32 + j;
            let v = match info.ggml_type {
                0 => f32::from_bits(get_u32(raw, idx * 4)),
                1 => f16_to_f32(get_u16(raw, idx * 2)),
                30 => f32::from_bits((get_u16(raw, idx * 2) as u32) << 16),
                t => return Err(format!("Unsupported ggml_type={} on tensor {}", t, name)),
            };
            vals[j] = v;
            amax = amax.max(v.abs());
        }
        let d_f16 = f32_to_f16_bits(amax / 127.0);
        let d_f32 = f16_to_f32(d_f16);
        let inv_d = if d_f32 > 0.0 { 1.0 / d_f32 } else { 0.0 };
        let dst = &mut q8[b * 34..(b + 1) * 34];
        dst[0..2].copy_from_slice(&d_f16.to_le_bytes());
        for j in 0..32 {
            let q = (vals[j] * inv_d).round().clamp(-127.0, 127.0) as i8;
            dst[2 + j] = q as u8;
        }
    }
    Ok((std::borrow::Cow::Owned(q8), 8, out_f, in_f))
}

pub fn precompute_freqs_cis_f32(rope_dim: usize, max_ctx: usize, theta: f32) -> Vec<f32> {
    let half = rope_dim / 2;
    let mut inv_freq = vec![0.0f32; half];
    for i in 0..half {
        inv_freq[i] = 1.0 / theta.powf((2 * i) as f32 / (rope_dim as f32));
    }
    let mut out = vec![0.0f32; max_ctx * rope_dim];
    for pos in 0..max_ctx {
        let p = pos as f32;
        let row = &mut out[pos * rope_dim..(pos + 1) * rope_dim];
        for i in 0..half {
            let ang = p * inv_freq[i];
            row[i] = ang.cos();
            row[half + i] = ang.sin();
        }
    }
    out
}

#[derive(Clone, Debug)]
pub struct QwenModelConfig {
    pub arch: String,
    pub model_name: String,
    pub num_layers: usize,
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub rope_theta: f32,
    pub norm_eps: f32,
    pub vocab_size: usize,
    pub max_ctx: usize,
    pub has_qk_norm: bool,
    pub thinking_template: bool,
    pub bos_id: Option<i32>,
    pub eos_id: i32,
    pub im_end_id: i32,
    pub stop_ids: HashSet<i32>,
}

impl QwenModelConfig {
    pub fn from_gguf(
        gguf: &GgufFile,
        tok: &BpeTokenizer,
        max_context_req: usize,
        fallback_stem: &str,
    ) -> Result<Self, String> {
        let arch = gguf
            .get_str("general.architecture")
            .ok_or("Missing general.architecture in GGUF")?
            .to_string();
        if arch == "inkling2_loop" {
            let n_phys = gguf.get_usize_or("inkling2_loop.block_count", 24);
            let l_start = gguf.get_usize_or("inkling2_loop.loop.start", 6);
            let l_end = gguf.get_usize_or("inkling2_loop.loop.end", 18);
            let l_cnt = gguf.get_usize_or("inkling2_loop.loop.count", 3);
            let logical_layers = n_phys + (l_cnt.saturating_sub(1)) * (l_end.saturating_sub(l_start));
            let n_heads = gguf
                .get_i32_arr("inkling2_loop.attention.head_count")
                .and_then(|v| v.first().copied())
                .map(|v| v as usize)
                .unwrap_or(24);
            let n_kv_heads = gguf
                .get_i32_arr("inkling2_loop.attention.head_count_kv")
                .and_then(|v| v.first().copied())
                .map(|v| v as usize)
                .unwrap_or(8);
            let model_name = gguf.get_str("general.name").unwrap_or("Inkling-2-Mini").to_string();
            let mut stop_ids = HashSet::new();
            stop_ids.insert(200006);
            stop_ids.insert(199999);
            let _ = tok;
            return Ok(Self {
                arch,
                model_name,
                num_layers: logical_layers,
                dim: gguf.get_usize_or("inkling2_loop.embedding_length", 2304),
                hidden_dim: gguf.get_usize_or("inkling2_loop.expert_feed_forward_length", 1152),
                n_heads,
                n_kv_heads,
                head_dim: gguf.get_usize_or("inkling2_loop.attention.key_length", 128),
                rope_dim: 0,
                rope_theta: 0.0,
                norm_eps: gguf.get_f32_or("inkling2_loop.attention.layer_norm_rms_epsilon", 1e-6),
                vocab_size: gguf.get_usize_or("inkling2_loop.vocab_size_unpadded", 200058),
                max_ctx: max_context_req.clamp(512, 8192),
                has_qk_norm: true,
                thinking_template: true,
                bos_id: None,
                eos_id: 200006,
                im_end_id: 200001,
                stop_ids,
            });
        }
        if !matches!(arch.as_str(), "qwen2" | "qwen3" | "llama") {
            return Err(format!("Unsupported GGUF architecture '{}'", arch));
        }

        let u_key = |suffix: &str| gguf.req_usize(&format!("{}.{}", arch, suffix));
        let f_key = |suffix: &str| gguf.req_f32(&format!("{}.{}", arch, suffix));

        let file_ctx = u_key("context_length")?;
        let max_ctx = max_context_req.min(file_ctx);
        let num_layers = u_key("block_count")?;
        let dim = u_key("embedding_length")?;
        let hidden_dim = u_key("feed_forward_length")?;
        let n_heads = u_key("attention.head_count")?;
        let n_kv_heads = u_key("attention.head_count_kv")?;
        let head_dim = gguf.get_usize_or(&format!("{}.attention.key_length", arch), dim / n_heads);
        let rope_dim = gguf.get_usize_or(&format!("{}.rope.dimension_count", arch), head_dim);
        let rope_theta = f_key("rope.freq_base")?;
        let norm_eps = f_key("attention.layer_norm_rms_epsilon")?;
        let vocab_size = gguf
            .get_str_arr("tokenizer.ggml.tokens")
            .ok_or("Missing tokenizer.ggml.tokens")?
            .len();
        let has_qk_norm = gguf.has_tensor("blk.0.attn_q_norm.weight");

        let model_name = gguf
            .get_str("general.name")
            .or_else(|| gguf.get_str("general.basename"))
            .unwrap_or(fallback_stem)
            .to_string();

        let chat_tmpl = gguf.get_str("tokenizer.chat_template").unwrap_or("");
        let lower_name = model_name.to_ascii_lowercase();
        let lower_stem = fallback_stem.to_ascii_lowercase();
        let is_embed = lower_name.contains("embed") || lower_stem.contains("embed");
        let is_pure_instruct = (lower_name.contains("instruct") || lower_stem.contains("instruct"))
            && !chat_tmpl.contains("enable_thinking")
            && !lower_name.contains("thinking")
            && !lower_stem.contains("thinking");
        let thinking_template = !is_embed
            && !is_pure_instruct
            && (chat_tmpl.contains("assistant\\n<think>\\n")
                || chat_tmpl.contains("assistant\n<think>\n")
                || chat_tmpl.contains("enable_thinking")
                || lower_name.contains("thinking")
                || lower_stem.contains("thinking"));

        let add_bos = gguf.get_bool_or("tokenizer.ggml.add_bos_token", true);
        let bos_id = if add_bos {
            gguf.get_u64("tokenizer.ggml.bos_token_id").map(|v| v as i32)
        } else {
            None
        };
        let eos_id = gguf.get_usize_or("tokenizer.ggml.eos_token_id", 151645) as i32;
        let im_end_id = tok.special_id("<|im_end|>").unwrap_or(eos_id);

        let mut stop_ids = HashSet::new();
        stop_ids.insert(eos_id);
        stop_ids.insert(im_end_id);
        for s in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>"] {
            if let Some(tid) = tok.special_id(s) {
                stop_ids.insert(tid);
            }
        }

        Ok(Self {
            arch,
            model_name,
            num_layers,
            dim,
            hidden_dim,
            n_heads,
            n_kv_heads,
            head_dim,
            rope_dim,
            rope_theta,
            norm_eps,
            vocab_size,
            max_ctx,
            has_qk_norm,
            thinking_template,
            bos_id,
            eos_id,
            im_end_id,
            stop_ids,
        })
    }
}
