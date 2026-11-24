//! Unified Persistent GPU Runtime Session, Unix-Socket Daemon, Client RPC, and OpenAI HTTP Server.

#![allow(dead_code)]

pub mod client;
pub mod daemon;
pub mod http;
pub mod session;

use std::sync::atomic::{AtomicBool, Ordering};

pub(crate) static SHUTDOWN_REQ: AtomicBool = AtomicBool::new(false);

extern "C" fn handle_sig(_sig: i32) {
    SHUTDOWN_REQ.store(true, Ordering::SeqCst);
}

extern "C" {
    fn signal(sig: i32, handler: extern "C" fn(i32)) -> usize;
}

const SIGHUP: i32 = 1;
const SIGINT: i32 = 2;
const SIGPIPE: i32 = 13;
const SIGTERM: i32 = 15;
const SIG_IGN: usize = 1;

pub(crate) fn install_signal_handlers_raw() {
    unsafe {
        signal(SIGINT, handle_sig);
        signal(SIGTERM, handle_sig);
        let _ = signal(SIGHUP, std::mem::transmute::<usize, extern "C" fn(i32)>(SIG_IGN));
        let _ = signal(SIGPIPE, std::mem::transmute::<usize, extern "C" fn(i32)>(SIG_IGN));
    }
}

pub(crate) fn json_escape(s: &str) -> String {
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

pub(crate) fn extract_json_str_after(s: &str) -> Option<String> {
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
