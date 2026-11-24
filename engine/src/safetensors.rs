//! Zero-Copy `mmap` Reader, Half-Precision Conversions, and Unified `.safetensors` Parser.

#![allow(dead_code)]

use std::collections::HashMap;
use std::fs::{self, File};
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::ptr;

use crate::hw::{get_u16, get_u32, get_u64};

const PROT_READ: i32 = 0x01;
const MAP_PRIVATE: i32 = 0x0002;
const MAP_FAILED_ISIZE: isize = -1;

extern "C" {
    fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut u8;
    fn munmap(addr: *mut u8, len: usize) -> i32;
}

pub struct MmapFile {
    ptr: *mut u8,
    len: usize,
}

unsafe impl Send for MmapFile {}
unsafe impl Sync for MmapFile {}

impl MmapFile {
    pub fn open(path: &Path) -> Result<Self, String> {
        let f = File::open(path).map_err(|e| format!("Failed to open {}: {}", path.display(), e))?;
        let len = f.metadata().map_err(|e| e.to_string())?.len() as usize;
        if len == 0 {
            return Err(format!("File {} is empty", path.display()));
        }
        let ptr = unsafe { mmap(ptr::null_mut(), len, PROT_READ, MAP_PRIVATE, f.as_raw_fd(), 0) };
        if (ptr as isize) == MAP_FAILED_ISIZE || ptr.is_null() {
            return Err(format!("mmap failed on {}", path.display()));
        }
        Ok(Self { ptr, len })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for MmapFile {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.len > 0 {
            unsafe {
                munmap(self.ptr, self.len);
            }
            self.ptr = ptr::null_mut();
        }
    }
}

// -----------------------------------------------------------------------------
// Half-Precision (`FP16` / `BF16`) Scalar Conversion Helpers
// -----------------------------------------------------------------------------

#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let mut exp = ((h >> 10) & 0x1f) as i32;
    let mut mant = (h & 0x03ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            exp = 1;
            while (mant & 0x0400) == 0 {
                mant <<= 1;
                exp -= 1;
            }
            mant &= 0x03ff;
            sign | (((exp + 127 - 15) as u32) << 23) | (mant << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | (((exp + 127 - 15) as u32) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

#[inline]
pub fn f32_to_f16_bits(v: f32) -> u16 {
    let f = v.to_bits();
    let sign = ((f >> 16) & 0x8000) as u16;
    let exp = ((f >> 23) & 0xff) as i32 - 127 + 15;
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let mant = (f & 0x007f_ffff) | 0x0080_0000;
        let shift = 14 - exp;
        let rounded = (mant + (1 << (shift - 1)) - 1 + ((mant >> shift) & 1)) >> shift;
        return sign | (rounded as u16);
    } else if exp >= 31 {
        return sign | 0x7c00;
    }
    let mant = f & 0x007f_ffff;
    let rounded = (mant + 0x0fff + ((mant >> 13) & 1)) >> 13;
    if rounded >= 0x0400 {
        if exp + 1 >= 31 {
            return sign | 0x7c00;
        }
        return sign | (((exp + 1) as u16) << 10);
    }
    sign | ((exp as u16) << 10) | (rounded as u16)
}

#[inline]
pub fn f32_to_bf16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    if (bits & 0x7f80_0000) == 0x7f80_0000 && (bits & 0x007f_ffff) != 0 {
        return ((bits >> 16) as u16) | 0x0040;
    }
    let lsb = (bits >> 16) & 1;
    let rounding_bias = 0x7fff + lsb;
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}

// -----------------------------------------------------------------------------
// Unified Zero-Crate `.safetensors` Reader & Shard Index Resolver
// -----------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct StSliceMeta {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub file_data_start: usize,
    pub byte_len: usize,
}

pub struct SafeTensorsFile {
    mmap: MmapFile,
    entries: HashMap<String, StSliceMeta>,
}

impl SafeTensorsFile {
    pub fn open(path: &Path) -> Result<Self, String> {
        let mmap = MmapFile::open(path)?;
        let raw = mmap.as_slice();
        if raw.len() < 8 {
            return Err(format!("Safetensors {} too small", path.display()));
        }
        let hdr_len = get_u64(raw, 0) as usize;
        if 8 + hdr_len > raw.len() {
            return Err(format!("Invalid safetensors header length {} in {}", hdr_len, path.display()));
        }
        let header_end = 8 + hdr_len;
        let hdr_str = std::str::from_utf8(&raw[8..header_end])
            .map_err(|e| format!("Invalid UTF-8 safetensors header in {}: {}", path.display(), e))?;
        let entries = parse_st_header(hdr_str, header_end)?;
        Ok(Self { mmap, entries })
    }

    pub fn tensor_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.entries.keys().cloned().collect();
        names.sort();
        names
    }

    #[inline]
    pub fn has_tensor(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    #[inline]
    pub fn meta(&self, name: &str) -> Option<&StSliceMeta> {
        self.entries.get(name)
    }

    pub fn tensor_slice(&self, name: &str) -> Result<&[u8], String> {
        let meta = self
            .entries
            .get(name)
            .ok_or_else(|| format!("Tensor '{}' not found in safetensors", name))?;
        Ok(&self.mmap.as_slice()[meta.file_data_start..meta.file_data_start + meta.byte_len])
    }

    pub fn raw_tensor_bytes(&self, name: &str) -> Result<(&[u8], StSliceMeta), String> {
        let meta = self
            .entries
            .get(name)
            .ok_or_else(|| format!("Tensor '{}' not found in safetensors shard", name))?
            .clone();
        let s = &self.mmap.as_slice()[meta.file_data_start..meta.file_data_start + meta.byte_len];
        Ok((s, meta))
    }

    pub fn decode_f32(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
        let meta = self
            .entries
            .get(name)
            .ok_or_else(|| format!("Tensor '{}' not found in safetensors", name))?;
        let s = &self.mmap.as_slice()[meta.file_data_start..meta.file_data_start + meta.byte_len];
        let nel: usize = meta.shape.iter().product();
        let mut out = Vec::with_capacity(nel);
        match meta.dtype.as_str() {
            "BF16" => {
                if s.len() != nel * 2 {
                    return Err(format!("BF16 tensor '{}' byte length mismatch", name));
                }
                for i in 0..nel {
                    out.push(f32::from_bits((get_u16(s, i * 2) as u32) << 16));
                }
            }
            "F16" => {
                if s.len() != nel * 2 {
                    return Err(format!("F16 tensor '{}' byte length mismatch", name));
                }
                for i in 0..nel {
                    out.push(f16_to_f32(get_u16(s, i * 2)));
                }
            }
            "F32" => {
                if s.len() != nel * 4 {
                    return Err(format!("F32 tensor '{}' byte length mismatch", name));
                }
                for i in 0..nel {
                    out.push(f32::from_bits(get_u32(s, i * 4)));
                }
            }
            other => return Err(format!("Unsupported safetensors dtype '{}' on '{}'", other, name)),
        }
        Ok((out, meta.shape.clone()))
    }
}

fn parse_st_header(hdr: &str, data_base: usize) -> Result<HashMap<String, StSliceMeta>, String> {
    let bytes = hdr.as_bytes();
    let mut i = 0usize;
    let skip_ws = |mut p: usize| -> usize {
        while p < bytes.len() && bytes[p].is_ascii_whitespace() {
            p += 1;
        }
        p
    };
    let read_qstr = |mut p: usize| -> Result<(String, usize), String> {
        p = skip_ws(p);
        if p >= bytes.len() || bytes[p] != b'"' {
            return Err("Expected '\"' in safetensors header".into());
        }
        p += 1;
        let mut s = String::new();
        while p < bytes.len() {
            let c = bytes[p];
            p += 1;
            if c == b'"' {
                return Ok((s, p));
            }
            if c == b'\\' && p < bytes.len() {
                s.push(bytes[p] as char);
                p += 1;
            } else {
                s.push(c as char);
            }
        }
        Err("Unterminated string in safetensors header".into())
    };

    i = skip_ws(i);
    if i >= bytes.len() || bytes[i] != b'{' {
        return Err("Safetensors header missing '{'".into());
    }
    i += 1;

    let mut map = HashMap::new();
    loop {
        i = skip_ws(i);
        if i >= bytes.len() || bytes[i] == b'}' {
            break;
        }
        if bytes[i] == b',' {
            i += 1;
            continue;
        }
        let (key, next_i) = read_qstr(i)?;
        i = skip_ws(next_i);
        if i >= bytes.len() || bytes[i] != b':' {
            return Err("Expected ':' in safetensors header".into());
        }
        i = skip_ws(i + 1);
        if i >= bytes.len() || bytes[i] != b'{' {
            return Err("Expected '{' for tensor object".into());
        }
        let mut depth = 1usize;
        let obj_start = i + 1;
        i += 1;
        while i < bytes.len() && depth > 0 {
            match bytes[i] {
                b'"' => {
                    i += 1;
                    while i < bytes.len() {
                        if bytes[i] == b'\\' {
                            i += 2;
                        } else if bytes[i] == b'"' {
                            i += 1;
                            break;
                        } else {
                            i += 1;
                        }
                    }
                }
                b'{' => {
                    depth += 1;
                    i += 1;
                }
                b'}' => {
                    depth -= 1;
                    i += 1;
                }
                _ => i += 1,
            }
        }
        let obj_str = &hdr[obj_start..i - 1];
        if key == "__metadata__" {
            continue;
        }
        let dtype = extract_str_val(obj_str, "\"dtype\"").unwrap_or_default();
        let shape = extract_usize_arr(obj_str, "\"shape\"").unwrap_or_default();
        let offs = extract_usize_arr(obj_str, "\"data_offsets\"").unwrap_or_default();
        if offs.len() == 2 && offs[1] >= offs[0] {
            map.insert(
                key,
                StSliceMeta {
                    dtype,
                    shape,
                    file_data_start: data_base + offs[0],
                    byte_len: offs[1] - offs[0],
                },
            );
        }
    }
    Ok(map)
}

fn extract_str_val(src: &str, key: &str) -> Option<String> {
    let idx = src.find(key)?;
    let rest = src[idx + key.len()..].trim_start().strip_prefix(':')?.trim_start().strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

fn extract_usize_arr(src: &str, key: &str) -> Option<Vec<usize>> {
    let idx = src.find(key)?;
    let rest = src[idx + key.len()..].trim_start().strip_prefix(':')?.trim_start().strip_prefix('[')?;
    let end = rest.find(']')?;
    let inner = rest[..end].trim();
    if inner.is_empty() {
        return Some(Vec::new());
    }
    let mut out = Vec::new();
    for p in inner.split(',') {
        out.push(p.trim().parse::<usize>().ok()?);
    }
    Some(out)
}

/// Parse `model.safetensors.index.json` -> `HashMap<tensor_name, shard_filename>`.
pub fn parse_safetensors_index(dir: &Path) -> Result<HashMap<String, String>, String> {
    let idx_path = dir.join("model.safetensors.index.json");
    if !idx_path.exists() {
        let single = dir.join("model.safetensors");
        if single.exists() {
            let st = SafeTensorsFile::open(&single)?;
            let mut m = HashMap::new();
            for k in st.tensor_names() {
                m.insert(k, "model.safetensors".to_string());
            }
            return Ok(m);
        }
        return Err(format!("Missing model.safetensors.index.json in {}", dir.display()));
    }
    let txt = fs::read_to_string(&idx_path).map_err(|e| e.to_string())?;
    let wm_idx = txt
        .find("\"weight_map\"")
        .ok_or("Missing weight_map in model.safetensors.index.json")?;
    let body = &txt[wm_idx + 12..];
    let open_brace = body.find('{').ok_or("Malformed weight_map")?;
    let sub = &body[open_brace + 1..];

    let mut map = HashMap::new();
    for line in sub.lines() {
        let ln = line.trim().trim_end_matches(',');
        if ln.starts_with('}') {
            break;
        }
        if let Some(colon) = ln.find(':') {
            let k = ln[..colon].trim().trim_matches('"');
            let v = ln[colon + 1..].trim().trim_matches('"');
            if !k.is_empty() && v.ends_with(".safetensors") {
                map.insert(k.to_string(), v.to_string());
            }
        }
    }
    Ok(map)
}
