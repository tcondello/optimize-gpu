//! Native macOS `aarch64-apple-darwin` POSIX Termios, Synchronized 2D TrueColor Cell Renderer,
//! Non-Blocking Keyboard/Mouse/Paste Input Parser, and `pbcopy` Helper.

use std::io::{self, Write};
use std::process::{Command, Stdio};

// =============================================================================
// Compiler-Verified macOS `aarch64-apple-darwin` POSIX Termios / Winsize / Poll ABI
// =============================================================================

const STDIN_FILENO: i32 = 0;
const STDOUT_FILENO: i32 = 1;

const TCSANOW: i32 = 0;
const NCCS: usize = 20;
const VMIN: usize = 16;
const VTIME: usize = 17;

const BRKINT: u64 = 0x0002;
const INPCK: u64 = 0x0010;
const ISTRIP: u64 = 0x0020;
const ICRNL: u64 = 0x0100;
const IXON: u64 = 0x0200;
const OPOST: u64 = 0x0001;
const CS8: u64 = 0x0300;
const ECHO: u64 = 0x0008;
const ICANON: u64 = 0x0100;
const ISIG: u64 = 0x0080;
const IEXTEN: u64 = 0x0400;

const TIOCGWINSZ: u64 = 0x4008_7468;
const POLLIN: i16 = 0x0001;

#[repr(C)]
#[derive(Clone, Copy)]
struct Termios {
    c_iflag: u64,
    c_oflag: u64,
    c_cflag: u64,
    c_lflag: u64,
    c_cc: [u8; NCCS],
    c_ispeed: u64,
    c_ospeed: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct WinSize {
    ws_row: u16,
    ws_col: u16,
    ws_xpixel: u16,
    ws_ypixel: u16,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PollFd {
    fd: i32,
    events: i16,
    revents: i16,
}

extern "C" {
    fn isatty(fd: i32) -> i32;
    fn tcgetattr(fd: i32, termios_p: *mut Termios) -> i32;
    fn tcsetattr(fd: i32, optional_actions: i32, termios_p: *const Termios) -> i32;
    fn ioctl(fd: i32, request: u64, ...) -> i32;
    fn poll(fds: *mut PollFd, nfds: u32, timeout: i32) -> i32;
    fn read(fd: i32, buf: *mut u8, count: usize) -> isize;
}

pub(super) struct RawTermGuard {
    orig: Termios,
    mouse_enabled: bool,
}

impl RawTermGuard {
    pub(super) fn enter() -> Result<Self, String> {
        if unsafe { isatty(STDIN_FILENO) == 0 || isatty(STDOUT_FILENO) == 0 } {
            return Err("rtx tui requires an interactive terminal (TTY)".into());
        }
        let mut orig = Termios {
            c_iflag: 0,
            c_oflag: 0,
            c_cflag: 0,
            c_lflag: 0,
            c_cc: [0; NCCS],
            c_ispeed: 0,
            c_ospeed: 0,
        };
        if unsafe { tcgetattr(STDIN_FILENO, &mut orig) } != 0 {
            return Err("tcgetattr failed on STDIN".into());
        }

        let mut raw = orig;
        raw.c_iflag &= !(BRKINT | ICRNL | INPCK | ISTRIP | IXON);
        raw.c_oflag &= !OPOST;
        raw.c_cflag |= CS8;
        raw.c_lflag &= !(ECHO | ICANON | IEXTEN | ISIG);
        raw.c_cc[VMIN] = 0;
        raw.c_cc[VTIME] = 0;

        if unsafe { tcsetattr(STDIN_FILENO, TCSANOW, &raw) } != 0 {
            return Err("tcsetattr failed entering raw mode".into());
        }

        // Install panic hook so terminal state is always restored cleanly
        let orig_copy = orig;
        let prev_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let _ = io::stdout().write_all(b"\x1b[?1006l\x1b[?1000l\x1b[?2004l\x1b[?25h\x1b[?1049l\x1b[0m");
            let _ = io::stdout().flush();
            unsafe {
                tcsetattr(STDIN_FILENO, TCSANOW, &orig_copy);
            }
            prev_hook(info);
        }));

        // Alternate screen + hide cursor + bracketed paste + SGR mouse wheel
        let mut out = io::stdout();
        out.write_all(b"\x1b[?1049h\x1b[?25l\x1b[?2004h\x1b[?1000h\x1b[?1006h")
            .map_err(|e| e.to_string())?;
        out.flush().map_err(|e| e.to_string())?;

        Ok(Self {
            orig,
            mouse_enabled: true,
        })
    }

    pub(super) fn set_mouse(&mut self, enable: bool) {
        if self.mouse_enabled == enable {
            return;
        }
        self.mouse_enabled = enable;
        let seq: &[u8] = if enable {
            b"\x1b[?1000h\x1b[?1006h"
        } else {
            b"\x1b[?1006l\x1b[?1000l"
        };
        let _ = io::stdout().write_all(seq);
        let _ = io::stdout().flush();
    }
}

impl Drop for RawTermGuard {
    fn drop(&mut self) {
        let _ = io::stdout().write_all(b"\x1b[?1006l\x1b[?1000l\x1b[?2004l\x1b[?25h\x1b[?1049l\x1b[0m");
        let _ = io::stdout().flush();
        unsafe {
            tcsetattr(STDIN_FILENO, TCSANOW, &self.orig);
        }
    }
}

pub(super) fn term_size() -> (usize, usize) {
    let mut ws = WinSize::default();
    if unsafe { ioctl(STDOUT_FILENO, TIOCGWINSZ, &mut ws) } == 0 && ws.ws_col >= 40 && ws.ws_row >= 12 {
        (ws.ws_col as usize, ws.ws_row as usize)
    } else {
        (100, 34)
    }
}

// =============================================================================
// TrueColor 2D Cell Buffer & Flicker-Free Synchronized Renderer
// =============================================================================

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct Rgb(pub u8, pub u8, pub u8);

// Base Markdown / Badge Palette Constants
pub(super) const BG_MAIN: Rgb      = Rgb(13, 17, 23);
pub(super) const BG_CODE: Rgb      = Rgb(14, 19, 32);
pub(super) const BG_CODE_HDR: Rgb  = Rgb(22, 30, 48);
pub(super) const BG_INLINE: Rgb    = Rgb(28, 36, 54);
pub(super) const BG_BADGE_ON: Rgb  = Rgb(20, 83, 45);
pub(super) const BG_BADGE_T1: Rgb  = Rgb(30, 58, 95);

pub(super) const FG_TEXT: Rgb      = Rgb(230, 237, 243);
pub(super) const FG_CYAN: Rgb      = Rgb(88, 204, 246);
pub(super) const FG_GREEN: Rgb     = Rgb(74, 222, 128);
pub(super) const FG_AMBER: Rgb     = Rgb(251, 191, 36);
pub(super) const FG_CORAL: Rgb     = Rgb(248, 113, 113);
pub(super) const FG_PURPLE: Rgb    = Rgb(192, 132, 252);
pub(super) const FG_CODE: Rgb      = Rgb(210, 228, 255);
pub(super) const FG_BORDER: Rgb    = Rgb(56, 84, 124);

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct Style {
    fg: Rgb,
    bg: Rgb,
    bold: bool,
    dim: bool,
}

impl Style {
    pub(super) const fn new(fg: Rgb, bg: Rgb) -> Self {
        Self { fg, bg, bold: false, dim: false }
    }
    pub(super) const fn bold(mut self) -> Self {
        self.bold = true;
        self
    }
    #[allow(dead_code)]
    pub(super) const fn dim(mut self) -> Self {
        self.dim = true;
        self
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct Cell {
    ch: char,
    style: Style,
    continuation: bool, // trailing half of a 2-column wide character
}

impl Default for Cell {
    fn default() -> Self {
        Self {
            ch: ' ',
            style: Style::new(FG_TEXT, BG_MAIN),
            continuation: false,
        }
    }
}

pub(super) fn char_display_width(c: char) -> usize {
    let u = c as u32;
    if u < 0x20 || (0x7f..0xa0).contains(&u) {
        return 0;
    }
    // Combining marks / zero-width joiners / variation selectors
    if (0x0300..=0x036F).contains(&u)
        || (0x200B..=0x200F).contains(&u)
        || (0xFE00..=0xFE0F).contains(&u)
    {
        return 0;
    }
    // Wide CJK / Hangul / Fullwidth / Emoji blocks
    if (0x1100..=0x115F).contains(&u)
        || (0x2E80..=0xA4CF).contains(&u)
        || (0xAC00..=0xD7A3).contains(&u)
        || (0xF900..=0xFAFF).contains(&u)
        || (0xFE10..=0xFE6F).contains(&u)
        || (0xFF01..=0xFF60).contains(&u)
        || (0x1F300..=0x1FAFF).contains(&u)
    {
        return 2;
    }
    1
}

pub(super) struct ScreenBuf {
    w: usize,
    h: usize,
    cells: Vec<Cell>,
    out_buf: Vec<u8>,
}

impl ScreenBuf {
    pub(super) fn new(w: usize, h: usize) -> Self {
        Self {
            w,
            h,
            cells: vec![Cell::default(); w * h],
            out_buf: Vec::with_capacity(w * h * 8),
        }
    }

    #[inline]
    pub(super) fn dims(&self) -> (usize, usize) {
        (self.w, self.h)
    }

    pub(super) fn resize_and_clear(&mut self, w: usize, h: usize, bg: Rgb) {
        self.w = w;
        self.h = h;
        let blank = Cell {
            ch: ' ',
            style: Style::new(FG_TEXT, bg),
            continuation: false,
        };
        self.cells.clear();
        self.cells.resize(w * h, blank);
    }

    pub(super) fn fill_row(&mut self, y: usize, bg: Rgb) {
        if y >= self.h {
            return;
        }
        let st = Style::new(FG_TEXT, bg);
        let row_start = y * self.w;
        for x in 0..self.w {
            self.cells[row_start + x] = Cell {
                ch: ' ',
                style: st,
                continuation: false,
            };
        }
    }

    pub(super) fn put_char(&mut self, x: usize, y: usize, ch: char, style: Style) -> usize {
        if y >= self.h || x >= self.w {
            return 0;
        }
        let cw = char_display_width(ch);
        if cw == 0 {
            return 0;
        }
        let idx = y * self.w + x;
        if cw == 2 {
            if x + 1 >= self.w {
                self.cells[idx] = Cell { ch: ' ', style, continuation: false };
                return 1;
            }
            self.cells[idx] = Cell { ch, style, continuation: false };
            self.cells[idx + 1] = Cell { ch: ' ', style, continuation: true };
            2
        } else {
            self.cells[idx] = Cell { ch, style, continuation: false };
            1
        }
    }

    pub(super) fn put_str(&mut self, mut x: usize, y: usize, max_x: usize, s: &str, style: Style) -> usize {
        let limit = max_x.min(self.w);
        for ch in s.chars() {
            let cw = char_display_width(ch);
            if cw == 0 {
                continue;
            }
            if x + cw > limit {
                break;
            }
            x += self.put_char(x, y, ch, style);
        }
        x
    }

    pub(super) fn render_ansi(&mut self, cursor: Option<(usize, usize)>) -> &[u8] {
        self.out_buf.clear();
        // Synchronized update start + hide cursor + home
        self.out_buf.extend_from_slice(b"\x1b[?2026h\x1b[?25l\x1b[H");

        let mut cur_style: Option<Style> = None;
        let mut utf8_tmp = [0u8; 4];

        for y in 0..self.h {
            if y > 0 {
                let _ = write!(self.out_buf, "\x1b[{};1H", y + 1);
            }
            let row = &self.cells[y * self.w..(y + 1) * self.w];
            for cell in row {
                if cell.continuation {
                    continue;
                }
                if cur_style != Some(cell.style) {
                    let s = cell.style;
                    let _ = write!(
                        self.out_buf,
                        "\x1b[0;{}{}38;2;{};{};{};48;2;{};{};{}m",
                        if s.bold { "1;" } else { "" },
                        if s.dim { "2;" } else { "" },
                        s.fg.0, s.fg.1, s.fg.2,
                        s.bg.0, s.bg.1, s.bg.2
                    );
                    cur_style = Some(s);
                }
                let encoded = cell.ch.encode_utf8(&mut utf8_tmp);
                self.out_buf.extend_from_slice(encoded.as_bytes());
            }
        }

        if let Some((cx, cy)) = cursor {
            let _ = write!(
                self.out_buf,
                "\x1b[{};{}H\x1b[?25h",
                (cy + 1).min(self.h),
                (cx + 1).min(self.w)
            );
        }
        self.out_buf.extend_from_slice(b"\x1b[0m\x1b[?2026l");
        &self.out_buf
    }
}

// =============================================================================
// Rich Markdown Styled Line Formatter (Code Cards, Inline Code, Bold, Headings)
// =============================================================================


#[derive(Debug, Clone)]
pub(super) enum InputEvent {
    Char(char),
    Paste(String),
    Enter,
    Newline,
    Backspace,
    Delete,
    Left,
    Right,
    Up,
    Down,
    Home,
    End,
    PageUp,
    PageDown,
    ScrollUp(usize),
    ScrollDown(usize),
    Esc,
    CtrlC,
    CtrlD,
    CtrlQ,
    CtrlL,
    CtrlR,
    CtrlS,
    CtrlT,
    CtrlY,
    CtrlW,
    CtrlU,
}

pub(super) fn poll_input_events(timeout_ms: i32, paste_acc: &mut Option<String>) -> Vec<InputEvent> {
    let mut pfd = PollFd {
        fd: STDIN_FILENO,
        events: POLLIN,
        revents: 0,
    };
    let rc = unsafe { poll(&mut pfd, 1, timeout_ms) };
    if rc <= 0 || (pfd.revents & POLLIN) == 0 {
        return Vec::new();
    }

    let mut raw = Vec::with_capacity(256);
    let mut tmp = [0u8; 512];
    loop {
        let n = unsafe { read(STDIN_FILENO, tmp.as_mut_ptr(), tmp.len()) };
        if n > 0 {
            raw.extend_from_slice(&tmp[..n as usize]);
            // Drain any additional immediately buffered bytes (e.g. paste burst)
            let mut p2 = PollFd { fd: STDIN_FILENO, events: POLLIN, revents: 0 };
            if unsafe { poll(&mut p2, 1, 0) } <= 0 || (p2.revents & POLLIN) == 0 {
                break;
            }
        } else {
            break;
        }
    }

    let mut events = Vec::new();
    let mut i = 0usize;
    while i < raw.len() {
        // Inside bracketed paste `\x1b[200~ ... \x1b[201~`
        if let Some(buf) = paste_acc.as_mut() {
            if raw[i..].starts_with(b"\x1b[201~") {
                let text = std::mem::take(buf).replace("\r\n", "\n").replace('\r', "\n");
                *paste_acc = None;
                events.push(InputEvent::Paste(text));
                i += 6;
                continue;
            }
            // Find next ESC or consume UTF-8 chunk
            let next_esc = raw[i + 1..].iter().position(|&b| b == 0x1b).map(|p| i + 1 + p).unwrap_or(raw.len());
            if raw[i] == 0x1b && raw.len() - i < 6 && b"\x1b[201~".starts_with(&raw[i..]) {
                // Wait briefly for remaining bracketed-paste terminator bytes
                let mut p3 = PollFd { fd: STDIN_FILENO, events: POLLIN, revents: 0 };
                if unsafe { poll(&mut p3, 1, 5) } > 0 {
                    let n = unsafe { read(STDIN_FILENO, tmp.as_mut_ptr(), tmp.len()) };
                    if n > 0 {
                        raw.extend_from_slice(&tmp[..n as usize]);
                        continue;
                    }
                }
            }
            buf.push_str(&String::from_utf8_lossy(&raw[i..next_esc]));
            i = next_esc;
            continue;
        }

        let b = raw[i];
        if b == 0x1b {
            // Escape or CSI / Alt sequence
            if i + 1 >= raw.len() {
                // Disambiguate standalone Esc from multi-byte CSI split
                let mut p3 = PollFd { fd: STDIN_FILENO, events: POLLIN, revents: 0 };
                if unsafe { poll(&mut p3, 1, 8) } > 0 {
                    let n = unsafe { read(STDIN_FILENO, tmp.as_mut_ptr(), tmp.len()) };
                    if n > 0 {
                        raw.extend_from_slice(&tmp[..n as usize]);
                    }
                }
                if i + 1 >= raw.len() {
                    events.push(InputEvent::Esc);
                    i += 1;
                    continue;
                }
            }

            let b1 = raw[i + 1];
            if b1 == b'\r' || b1 == b'\n' {
                events.push(InputEvent::Newline); // Alt+Enter!
                i += 2;
                continue;
            }
            if b1 == b'[' {
                if raw[i..].starts_with(b"\x1b[200~") {
                    *paste_acc = Some(String::new());
                    i += 6;
                    continue;
                }
                // SGR Mouse: `\x1b[<btn;col;rowM` or `m`
                if i + 2 < raw.len() && raw[i + 2] == b'<' {
                    let mut end = i + 3;
                    while end < raw.len() && raw[end] != b'M' && raw[end] != b'm' {
                        end += 1;
                    }
                    if end < raw.len() {
                        if let Ok(s) = std::str::from_utf8(&raw[i + 3..end]) {
                            let mut parts = s.split(';');
                            if let Some(Ok(btn)) = parts.next().map(|p| p.parse::<u32>()) {
                                if btn == 64 {
                                    events.push(InputEvent::ScrollUp(3));
                                } else if btn == 65 {
                                    events.push(InputEvent::ScrollDown(3));
                                }
                            }
                        }
                        i = end + 1;
                        continue;
                    }
                }
                // Standard CSI sequence `\x1b[...<final>`
                let mut end = i + 2;
                while end < raw.len() && !(0x40..=0x7e).contains(&raw[end]) {
                    end += 1;
                }
                if end < raw.len() {
                    let params = std::str::from_utf8(&raw[i + 2..end]).unwrap_or("");
                    let fin = raw[end];
                    match (params, fin) {
                        ("", b'A') => events.push(InputEvent::Up),
                        ("", b'B') => events.push(InputEvent::Down),
                        ("", b'C') => events.push(InputEvent::Right),
                        ("", b'D') => events.push(InputEvent::Left),
                        ("", b'H') | ("1", b'~') | ("7", b'~') => events.push(InputEvent::Home),
                        ("", b'F') | ("4", b'~') | ("8", b'~') => events.push(InputEvent::End),
                        ("3", b'~') => events.push(InputEvent::Delete),
                        ("5", b'~') => events.push(InputEvent::PageUp),
                        ("6", b'~') => events.push(InputEvent::PageDown),
                        // Kitty / CSI-u Shift+Enter or Alt+Enter (`13;2u`, `13;3u`)
                        ("13;2", b'u') | ("13;3", b'u') | ("27;2;13", b'~') => {
                            events.push(InputEvent::Newline);
                        }
                        _ => {}
                    }
                    i = end + 1;
                    continue;
                }
            } else if b1 == b'O' && i + 2 < raw.len() {
                match raw[i + 2] {
                    b'H' => events.push(InputEvent::Home),
                    b'F' => events.push(InputEvent::End),
                    b'A' => events.push(InputEvent::Up),
                    b'B' => events.push(InputEvent::Down),
                    b'C' => events.push(InputEvent::Right),
                    b'D' => events.push(InputEvent::Left),
                    _ => {}
                }
                i += 3;
                continue;
            }
            events.push(InputEvent::Esc);
            i += 1;
            continue;
        }

        match b {
            0x03 => events.push(InputEvent::CtrlC),
            0x04 => events.push(InputEvent::CtrlD),
            0x11 => events.push(InputEvent::CtrlQ),
            0x0c => events.push(InputEvent::CtrlL),
            0x12 => events.push(InputEvent::CtrlR),
            0x13 => events.push(InputEvent::CtrlS),
            0x14 => events.push(InputEvent::CtrlT),
            0x19 => events.push(InputEvent::CtrlY),
            0x17 => events.push(InputEvent::CtrlW),
            0x15 => events.push(InputEvent::CtrlU),
            0x01 => events.push(InputEvent::Home), // Ctrl+A
            0x05 => events.push(InputEvent::End),  // Ctrl+E
            0x0a => events.push(InputEvent::Newline), // Ctrl+J
            0x0d => events.push(InputEvent::Enter),   // Enter (\r)
            0x7f | 0x08 => events.push(InputEvent::Backspace),
            0x09 => {
                for _ in 0..4 {
                    events.push(InputEvent::Char(' '));
                }
            }
            c if c >= 0x20 => {
                // Decode UTF-8 scalar
                let width = if c < 0x80 {
                    1
                } else if (c & 0xe0) == 0xc0 {
                    2
                } else if (c & 0xf0) == 0xe0 {
                    3
                } else {
                    4
                };
                let end = (i + width).min(raw.len());
                if let Ok(s) = std::str::from_utf8(&raw[i..end]) {
                    if let Some(ch) = s.chars().next() {
                        events.push(InputEvent::Char(ch));
                    }
                }
                i = end;
                continue;
            }
            _ => {}
        }
        i += 1;
    }
    events
}

// =============================================================================
// Chat State, Telemetry Parser, & Clipboard Helper
// =============================================================================


pub(super) fn copy_to_macos_clipboard(text: &str) -> Result<(), String> {
    let mut child = Command::new("/usr/bin/pbcopy")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("pbcopy failed: {}", e))?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(text.as_bytes()).map_err(|e| e.to_string())?;
    }
    let _ = child.wait();
    Ok(())
}

