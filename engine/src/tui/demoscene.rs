//! Informative 1990s Demoscene × Charm (`Lip Gloss`) Telemetry HUD, Palettes, Audio Spectrum,
//! Slash-Command Palette, and Opt-In `/demo` Easter-Egg Showcase.
//!
//! Design Principle: **Informative, Never Performative.**
//! - `0` gratuitous idle animations (`0.0%` CPU when GPU & audio are idle).
//! - Every gradient, bar, and sparkline encodes real hardware or session state:
//!   1. Segmented GDDR6X VRAM copper bar (`Thinker`, `Spec/LoRA`, `Voice Arena`, `Free`).
//!   2. Real `tok/s` & `% of 760.3 GB/s GDDR6X roofline` sparkline across turns.
//!   3. Real 16-band Hann-Goertzel audio spectrum active *only* during live mic/TTS/synth PCM.
//!   4. Interactive `/` command palette & toggleable right-side Telemetry Inspector (`Ctrl+B`).

use std::f32::consts::TAU;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use super::telemetry::HwTelemetry;
use super::terminal::{Rgb, ScreenBuf, Style, FG_AMBER, FG_CORAL, FG_CYAN, FG_GREEN};
use crate::voice::audio::{CoreAudioPlayer, TTS_SAMPLE_RATE};

const NUM_STARS: usize = 160;
const NUM_BANDS: usize = 16;
const GDDR6X_PEAK_GBS: f64 = 760.3;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum DemoTheme {
    CharmCopper,
    SecondReality,
    ImpulseTracker,
    AmigaDusk,
}

impl DemoTheme {
    pub(super) fn next(self) -> Self {
        match self {
            Self::CharmCopper => Self::SecondReality,
            Self::SecondReality => Self::ImpulseTracker,
            Self::ImpulseTracker => Self::AmigaDusk,
            Self::AmigaDusk => Self::CharmCopper,
        }
    }

    pub(super) fn from_arg(arg: &str) -> Option<Self> {
        match arg.trim().to_ascii_lowercase().as_str() {
            "1" | "charm" | "crush" | "charm_copper" | "default" => Some(Self::CharmCopper),
            "2" | "sr" | "second_reality" | "secondreality" | "fc" => Some(Self::SecondReality),
            "3" | "it" | "ft2" | "tracker" | "impulse" | "impulse_tracker" => {
                Some(Self::ImpulseTracker)
            }
            "4" | "amiga" | "copper" | "dusk" | "amiga_dusk" | "aga" => Some(Self::AmigaDusk),
            _ => None,
        }
    }

    pub(super) fn label(self) -> &'static str {
        match self {
            Self::CharmCopper => "Charm Copper",
            Self::SecondReality => "Second Reality",
            Self::ImpulseTracker => "Impulse Tracker",
            Self::AmigaDusk => "Amiga Dusk",
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct Palette {
    pub bg_void: Rgb,
    pub bg_header: Rgb,
    pub bg_surface: Rgb,
    pub bg_deck: Rgb,
    pub bg_footer: Rgb,
    pub border_dim: Rgb,
    pub border_mid: Rgb,
    pub border_hi: Rgb,
    pub grad_a: Rgb,
    pub grad_b: Rgb,
    pub grad_c: Rgb,
    pub text_hi: Rgb,
    pub text_muted: Rgb,
    pub text_dim: Rgb,
    pub accent_user: Rgb,
    pub accent_bot: Rgb,
}

pub(super) fn theme_palette(theme: DemoTheme) -> Palette {
    match theme {
        // Inspired by charm.land (Crush / Lip Gloss): soft obsidian surfaces + indigo -> cyan -> warm copper
        DemoTheme::CharmCopper => Palette {
            bg_void: Rgb(11, 13, 20),
            bg_header: Rgb(16, 19, 30),
            bg_surface: Rgb(18, 22, 35),
            bg_deck: Rgb(15, 18, 28),
            bg_footer: Rgb(14, 17, 26),
            border_dim: Rgb(36, 44, 66),
            border_mid: Rgb(62, 76, 114),
            border_hi: Rgb(134, 120, 246),
            grad_a: Rgb(134, 116, 249), // Charm Violet-Indigo
            grad_b: Rgb(76, 214, 248),  // Lip-Gloss Cyan
            grad_c: Rgb(252, 162, 108), // Amiga Warm Copper
            text_hi: Rgb(232, 238, 248),
            text_muted: Rgb(138, 152, 182),
            text_dim: Rgb(82, 94, 122),
            accent_user: Rgb(76, 214, 248),
            accent_bot: Rgb(252, 162, 108),
        },
        DemoTheme::SecondReality => Palette {
            bg_void: Rgb(8, 11, 20),
            bg_header: Rgb(13, 18, 34),
            bg_surface: Rgb(15, 22, 40),
            bg_deck: Rgb(11, 16, 30),
            bg_footer: Rgb(12, 17, 31),
            border_dim: Rgb(28, 48, 84),
            border_mid: Rgb(52, 96, 158),
            border_hi: Rgb(72, 220, 255),
            grad_a: Rgb(0, 224, 255),
            grad_b: Rgb(238, 72, 188),
            grad_c: Rgb(255, 206, 68),
            text_hi: Rgb(234, 242, 255),
            text_muted: Rgb(130, 156, 194),
            text_dim: Rgb(74, 96, 130),
            accent_user: Rgb(72, 220, 255),
            accent_bot: Rgb(255, 206, 68),
        },
        DemoTheme::ImpulseTracker => Palette {
            bg_void: Rgb(11, 14, 18),
            bg_header: Rgb(18, 24, 31),
            bg_surface: Rgb(20, 27, 35),
            bg_deck: Rgb(15, 20, 26),
            bg_footer: Rgb(16, 22, 28),
            border_dim: Rgb(38, 54, 64),
            border_mid: Rgb(68, 98, 112),
            border_hi: Rgb(244, 204, 92),
            grad_a: Rgb(78, 228, 136),
            grad_b: Rgb(244, 204, 92),
            grad_c: Rgb(82, 196, 228),
            text_hi: Rgb(230, 238, 232),
            text_muted: Rgb(138, 162, 158),
            text_dim: Rgb(82, 102, 100),
            accent_user: Rgb(82, 196, 228),
            accent_bot: Rgb(78, 228, 136),
        },
        DemoTheme::AmigaDusk => Palette {
            bg_void: Rgb(13, 10, 22),
            bg_header: Rgb(21, 16, 36),
            bg_surface: Rgb(24, 18, 42),
            bg_deck: Rgb(17, 13, 30),
            bg_footer: Rgb(18, 14, 32),
            border_dim: Rgb(54, 38, 76),
            border_mid: Rgb(106, 68, 132),
            border_hi: Rgb(250, 166, 82),
            grad_a: Rgb(250, 128, 56),
            grad_b: Rgb(255, 222, 138),
            grad_c: Rgb(176, 98, 250),
            text_hi: Rgb(244, 236, 230),
            text_muted: Rgb(168, 146, 184),
            text_dim: Rgb(98, 82, 118),
            accent_user: Rgb(255, 214, 128),
            accent_bot: Rgb(250, 142, 72),
        },
    }
}

#[inline]
fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    let af = a as f32;
    let bf = b as f32;
    (af + (bf - af) * t.clamp(0.0, 1.0)).round() as u8
}

#[inline]
pub(super) fn lerp_rgb(a: Rgb, b: Rgb, t: f32) -> Rgb {
    Rgb(lerp_u8(a.0, b.0, t), lerp_u8(a.1, b.1, t), lerp_u8(a.2, b.2, t))
}

#[inline]
fn scale_rgb(c: Rgb, k: f32) -> Rgb {
    let s = k.clamp(0.0, 2.0);
    Rgb(
        ((c.0 as f32) * s).min(255.0) as u8,
        ((c.1 as f32) * s).min(255.0) as u8,
        ((c.2 as f32) * s).min(255.0) as u8,
    )
}

pub(super) fn copper_gradient(pal: &Palette, phase: f32) -> Rgb {
    let u = phase.rem_euclid(1.0);
    if u < 0.5 {
        lerp_rgb(pal.grad_a, pal.grad_b, u * 2.0)
    } else {
        lerp_rgb(pal.grad_b, pal.grad_c, (u - 0.5) * 2.0)
    }
}

fn copper_loop(pal: &Palette, phase: f32) -> Rgb {
    let u = phase.rem_euclid(1.0);
    if u < 0.333 {
        lerp_rgb(pal.grad_a, pal.grad_b, u * 3.0)
    } else if u < 0.666 {
        lerp_rgb(pal.grad_b, pal.grad_c, (u - 0.333) * 3.0)
    } else {
        lerp_rgb(pal.grad_c, pal.grad_a, (u - 0.666) * 3.0)
    }
}

#[derive(Clone, Copy)]
struct Star3D {
    x: f32,
    y: f32,
    z: f32,
}

pub(super) struct TelemetrySnapshot<'a> {
    pub model_name: &'a str,
    pub is_inkling: bool,
    pub gguf_size_gb: f64,
    pub active_gb_per_tok: f64,
    pub active_sms: u32,
    pub ping_ms: f64,
    pub daemon_online: bool,
    pub effort: &'a str,
    pub spec_on: bool,
    pub gamma: usize,
    pub temp: f32,
    pub max_tokens: usize,
    pub ctx_used: usize,
    pub max_ctx: usize,
    pub cached_prefix: usize,
    pub active_tps: f64,
    pub last_ttft_ms: f64,
    pub last_prefill_tps: f64,
    pub last_launches: usize,
    pub last_spec_extra: usize,
    pub think_words: usize,
    pub answer_words: usize,
    pub expand_thinking: bool,
    pub voice_enabled: bool,
    pub hands_free: bool,
    pub voice_speaker: &'a str,
    pub voice_model_ref: &'a str,
    pub last_asr_ms: f64,
    pub last_ttfa_ms: f64,
    pub is_recording: bool,
    pub is_transcribing: bool,
    pub is_speaking: bool,
    pub is_streaming: bool,
    pub hw: Option<&'a HwTelemetry>,
}

pub(super) struct SlashItem {
    pub cmd: &'static str,
    pub args: &'static str,
    pub desc: &'static str,
    pub current: String,
}

pub(super) fn filter_slash_commands(
    prefix: &str,
    snap: &TelemetrySnapshot<'_>,
    theme_name: &str,
    sidebar_on: bool,
) -> Vec<SlashItem> {
    let q = prefix.trim().to_ascii_lowercase();
    if !q.starts_with('/') || q.contains(char::is_whitespace) {
        return Vec::new();
    }

    let mut all = Vec::with_capacity(18);
    if snap.is_inkling {
        all.push(SlashItem {
            cmd: "/effort",
            args: "<0|0.1|0.2|0.7|0.9>",
            desc: "Thinking depth (0 = instant ~122 tok/s)",
            current: format!("now: {}", snap.effort),
        });
    }
    all.push(SlashItem {
        cmd: "/fold",
        args: "",
        desc: "Expand or collapse <think> traces (Ctrl+O)",
        current: if snap.expand_thinking { "expanded".into() } else { "collapsed".into() },
    });
    all.push(SlashItem {
        cmd: "/inspector",
        args: "[on|off|auto]",
        desc: "Toggle right-side VRAM & Roofline sidebar (Ctrl+B)",
        current: if sidebar_on { "shown".into() } else { "hidden".into() },
    });
    all.push(SlashItem {
        cmd: "/listen",
        args: "",
        desc: "Capture Mac mic -> RTX 3080 STT -> Voice reply (Ctrl+R)",
        current: format!("ASR-{}", snap.voice_model_ref),
    });
    all.push(SlashItem {
        cmd: "/talk",
        args: "[on|off]",
        desc: "Hands-free continuous voice conversation",
        current: if snap.hands_free { "ON".into() } else { "OFF".into() },
    });
    all.push(SlashItem {
        cmd: "/speak",
        args: "[on|off]",
        desc: "Toggle automatic 24 kHz spoken replies",
        current: if snap.voice_enabled { "ON".into() } else { "OFF".into() },
    });
    all.push(SlashItem {
        cmd: "/voice",
        args: "<speaker>",
        desc: "Select TTS voice (ryan, serena, vivian, aiden...)",
        current: snap.voice_speaker.to_string(),
    });
    all.push(SlashItem {
        cmd: "/spec",
        args: "[on|off]",
        desc: "Toggle lossless speculative verification (Ctrl+S)",
        current: if snap.spec_on { "ON".into() } else { "OFF".into() },
    });
    if snap.is_inkling {
        all.push(SlashItem {
            cmd: "/gamma",
            args: "<1..7>",
            desc: "Set DSpark block-8 draft length",
            current: format!("γ={}", snap.gamma),
        });
    } else {
        all.push(SlashItem {
            cmd: "/rag",
            args: "[wiki|<qvec>|off]",
            desc: "Same-backbone GPU VectorDB retrieval",
            current: String::new(),
        });
        all.push(SlashItem {
            cmd: "/topic",
            args: "[wiki|list|off]",
            desc: "Sharded Topic Parametric LoRA router",
            current: String::new(),
        });
        all.push(SlashItem {
            cmd: "/lora",
            args: "<path|off>",
            desc: "Hot-swap Rank-r PEFT LoRA adapter (<1 ms)",
            current: String::new(),
        });
    }
    all.push(SlashItem {
        cmd: "/theme",
        args: "[charm|sr|it|amiga]",
        desc: "Switch TUI palette (Ctrl+T)",
        current: theme_name.to_string(),
    });
    all.push(SlashItem {
        cmd: "/bench",
        args: if snap.is_inkling { "[spec]" } else { "[t1]" },
        desc: "Run live RTX 3080 throughput benchmark",
        current: if snap.active_tps > 0.0 { format!("{:.1} tok/s", snap.active_tps) } else { String::new() },
    });
    all.push(SlashItem {
        cmd: "/verify",
        args: "",
        desc: "Inspect loaded GGUF file, size, & architecture",
        current: format!("{:.2} GB", snap.gguf_size_gb),
    });
    all.push(SlashItem {
        cmd: "/copy",
        args: "[code]",
        desc: "Copy last reply or code block to clipboard (Ctrl+Y)",
        current: String::new(),
    });
    all.push(SlashItem {
        cmd: "/clear",
        args: "",
        desc: "Clear chat & reset VRAM KV+ShortConv cache (Ctrl+L)",
        current: String::new(),
    });
    all.push(SlashItem {
        cmd: "/demo",
        args: "[1..4]",
        desc: "Launch opt-in 1990s 60 FPS Megademo Easter egg",
        current: String::new(),
    });
    all.push(SlashItem {
        cmd: "/quit",
        args: "",
        desc: "Exit TUI keeping GPU daemon warm (Ctrl+Q)",
        current: String::new(),
    });

    all.into_iter().filter(|it| it.cmd.starts_with(&q)).collect()
}

pub(super) struct DemosceneEngine {
    t0: Instant,
    last_step: Instant,
    time_s: f32,
    theme: DemoTheme,
    sidebar_pref: Option<bool>, // None = auto (>= 124 cols), Some(b) = user-pinned

    showcase_open: bool,
    showcase_scene: usize,
    showcase_auto: bool,
    showcase_scene_t0: Instant,

    stars: [Star3D; NUM_STARS],
    rng_state: u64,

    bands: [f32; NUM_BANDS],
    peaks: [f32; NUM_BANDS],
    peak_vel: [f32; NUM_BANDS],
    audio_active: bool,

    tps_history: Vec<f64>,

    fire_w: usize,
    fire_h: usize,
    fire_buf: Vec<u8>,
    pix_w: usize,
    pix_h: usize,
    pixels: Vec<Rgb>,

    mod_playing: Arc<AtomicBool>,
    mod_cancel: Option<Arc<AtomicBool>>,
}

impl DemosceneEngine {
    pub(super) fn new() -> Self {
        let now = Instant::now();
        let mut eng = Self {
            t0: now,
            last_step: now,
            time_s: 0.0,
            theme: DemoTheme::CharmCopper,
            sidebar_pref: None,
            showcase_open: false,
            showcase_scene: 0,
            showcase_auto: true,
            showcase_scene_t0: now,
            stars: [Star3D { x: 0.0, y: 0.0, z: 1.0 }; NUM_STARS],
            rng_state: 0x9E37_79B9_7F4A_7C15,
            bands: [0.0; NUM_BANDS],
            peaks: [0.0; NUM_BANDS],
            peak_vel: [0.0; NUM_BANDS],
            audio_active: false,
            tps_history: Vec::with_capacity(32),
            fire_w: 0,
            fire_h: 0,
            fire_buf: Vec::new(),
            pix_w: 0,
            pix_h: 0,
            pixels: Vec::new(),
            mod_playing: Arc::new(AtomicBool::new(false)),
            mod_cancel: None,
        };
        for i in 0..NUM_STARS {
            eng.stars[i] = eng.spawn_star(true);
        }
        eng
    }

    fn next_rand_f32(&mut self) -> f32 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        ((x >> 40) as u32 as f32) / ((1u32 << 24) as f32)
    }

    fn spawn_star(&mut self, random_z: bool) -> Star3D {
        let x = (self.next_rand_f32() * 2.0 - 1.0) * 1.6;
        let y = (self.next_rand_f32() * 2.0 - 1.0) * 1.1;
        let z = if random_z {
            0.08 + self.next_rand_f32() * 1.92
        } else {
            1.95 + self.next_rand_f32() * 0.15
        };
        Star3D { x, y, z }
    }

    pub(super) fn palette(&self) -> Palette {
        theme_palette(self.theme)
    }

    pub(super) fn sidebar_visible(&self, term_w: usize) -> bool {
        if term_w < 100 {
            return false;
        }
        self.sidebar_pref.unwrap_or(term_w >= 124)
    }

    pub(super) fn state_flags(&self) -> (&'static str, bool, bool) {
        (
            self.theme.label(),
            self.showcase_open,
            self.mod_playing.load(Ordering::Relaxed),
        )
    }

    pub(super) fn record_turn_tps(&mut self, tps: f64) {
        if tps > 0.5 {
            if self.tps_history.len() >= 24 {
                self.tps_history.remove(0);
            }
            self.tps_history.push(tps);
        }
    }

    pub(super) fn configure(
        &mut self,
        cmd: &str,
        arg: &str,
        term_w: usize,
        viz_pcm: &Arc<Mutex<Vec<f32>>>,
    ) -> String {
        match cmd {
            "theme_cycle" => {
                self.theme = self.theme.next();
                format!("Palette: {}", self.theme.label())
            }
            "theme" => {
                if arg.is_empty() {
                    self.theme = self.theme.next();
                } else if let Some(t) = DemoTheme::from_arg(arg) {
                    self.theme = t;
                } else {
                    return "Themes: charm (default), sr (Second Reality), it (Impulse Tracker), amiga (Amiga Dusk)".into();
                }
                format!("Palette: {}", self.theme.label())
            }
            "sidebar" => {
                let cur = self.sidebar_visible(term_w);
                let next = match arg.trim().to_ascii_lowercase().as_str() {
                    "on" | "1" | "show" => true,
                    "off" | "0" | "hide" => false,
                    "auto" => {
                        self.sidebar_pref = None;
                        return "Telemetry sidebar set to auto (>= 124 cols)".into();
                    }
                    _ => !cur,
                };
                self.sidebar_pref = Some(next);
                if next {
                    "Telemetry Inspector: shown (Ctrl+B to hide)".into()
                } else {
                    "Telemetry Inspector: hidden (Ctrl+B to show)".into()
                }
            }
            "demo_open" => {
                self.showcase_open = !self.showcase_open;
                self.showcase_scene_t0 = Instant::now();
                if let Ok(n) = arg.trim().parse::<usize>() {
                    if (1..=4).contains(&n) {
                        self.showcase_open = true;
                        self.showcase_scene = n - 1;
                        self.showcase_auto = false;
                    }
                } else if arg == "off" {
                    self.showcase_open = false;
                }
                if self.showcase_open {
                    "Entered 90s Megademo (1-4/Space: scene · T: palette · M: synth · Esc: back to chat)".into()
                } else {
                    "Returned to chat".into()
                }
            }
            "demo_scene" => {
                if arg == "next" {
                    self.showcase_scene = (self.showcase_scene + 1) % 4;
                    self.showcase_auto = false;
                } else if let Ok(n) = arg.parse::<usize>() {
                    if (1..=4).contains(&n) {
                        self.showcase_scene = n - 1;
                        self.showcase_auto = false;
                    }
                }
                self.showcase_scene_t0 = Instant::now();
                String::new()
            }
            "mod" => {
                let is_on = self.mod_playing.load(Ordering::Relaxed);
                let want_on = match arg.trim().to_ascii_lowercase().as_str() {
                    "on" | "1" | "play" | "start" => true,
                    "off" | "0" | "stop" => false,
                    _ => !is_on,
                };
                if want_on && !is_on {
                    self.start_mod_synth(Arc::clone(viz_pcm));
                    "Playing 4-ch Amiga tracker synth (/mod off or Esc stops)".into()
                } else if !want_on && is_on {
                    self.stop_mod_synth();
                    "Stopped tracker synth".into()
                } else {
                    "Tracker synth already active".into()
                }
            }
            "stop_all" => {
                self.showcase_open = false;
                self.stop_mod_synth();
                String::new()
            }
            _ => String::new(),
        }
    }

    fn start_mod_synth(&mut self, viz_pcm: Arc<Mutex<Vec<f32>>>) {
        self.stop_mod_synth();
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_bg = Arc::clone(&cancel);
        let playing_bg = Arc::clone(&self.mod_playing);
        playing_bg.store(true, Ordering::Relaxed);
        self.mod_cancel = Some(cancel);

        thread::spawn(move || {
            let Ok(mut player) = CoreAudioPlayer::new(TTS_SAMPLE_RATE) else {
                playing_bg.store(false, Ordering::Relaxed);
                return;
            };
            run_tracker_chiptune_loop(&mut player, &cancel_bg, &viz_pcm);
            player.abort_immediate();
            playing_bg.store(false, Ordering::Relaxed);
        });
    }

    fn stop_mod_synth(&mut self) {
        if let Some(c) = self.mod_cancel.take() {
            c.store(true, Ordering::Relaxed);
        }
        self.mod_playing.store(false, Ordering::Relaxed);
    }

    /// Advances active animations ONLY when real work or audio is occurring (`0.0%` idle CPU!).
    pub(super) fn step(
        &mut self,
        pcm_window: Option<(&[f32], u32)>,
        gpu_busy: bool,
    ) -> bool {
        let now = Instant::now();
        let dt = now.duration_since(self.last_step).as_secs_f32().clamp(0.001, 0.10);
        self.last_step = now;

        let mod_on = self.mod_playing.load(Ordering::Relaxed);
        let has_pcm = pcm_window.map_or(false, |(s, _)| s.len() >= 64);

        if has_pcm {
            let (samples, sr) = pcm_window.unwrap();
            self.update_spectrum_from_pcm(samples, sr);
            self.audio_active = true;
        } else if self.audio_active {
            // Smoothly decay audio spectrum bars back to calm zero once speech ends
            let mut any_left = false;
            for i in 0..NUM_BANDS {
                self.bands[i] *= 0.72;
                self.peak_vel[i] += dt * 0.9;
                self.peaks[i] = (self.peaks[i] - self.peak_vel[i] * dt * 3.0).max(self.bands[i]);
                if self.bands[i] > 0.015 || self.peaks[i] > 0.02 {
                    any_left = true;
                } else {
                    self.bands[i] = 0.0;
                    self.peaks[i] = 0.0;
                    self.peak_vel[i] = 0.0;
                }
            }
            if !any_left {
                self.audio_active = false;
            }
        }

        let need_anim = self.showcase_open || mod_on || gpu_busy || has_pcm || self.audio_active;
        if !need_anim {
            return false;
        }

        self.time_s = self.t0.elapsed().as_secs_f32();

        if self.showcase_open {
            for i in 0..NUM_STARS {
                self.stars[i].z -= dt * 0.85;
                if self.stars[i].z <= 0.06 {
                    self.stars[i] = self.spawn_star(false);
                }
            }
            if self.showcase_auto && self.showcase_scene_t0.elapsed() > Duration::from_secs(11) {
                self.showcase_scene = (self.showcase_scene + 1) % 4;
                self.showcase_scene_t0 = now;
            }
        }

        true
    }

    fn update_spectrum_from_pcm(&mut self, samples: &[f32], sample_rate: u32) {
        let win_len = samples.len().min(256);
        let tail = &samples[samples.len() - win_len..];
        let sr = (sample_rate.max(8_000)) as f32;
        let n_inv = 1.0 / (win_len as f32);

        for b in 0..NUM_BANDS {
            let frac = (b as f32) / ((NUM_BANDS - 1) as f32);
            let freq = 85.0 * (6400.0f32 / 85.0).powf(frac);
            if freq >= sr * 0.47 {
                self.bands[b] *= 0.80;
                continue;
            }
            let omega = TAU * freq / sr;
            let coeff = 2.0 * omega.cos();
            let mut s1 = 0.0f32;
            let mut s2 = 0.0f32;
            for (n, &smp) in tail.iter().enumerate() {
                let hann = 0.5 - 0.5 * (TAU * (n as f32) * n_inv).cos();
                let s0 = smp * hann + coeff * s1 - s2;
                s2 = s1;
                s1 = s0;
            }
            let power = (s1 * s1 + s2 * s2 - coeff * s1 * s2).max(0.0);
            let mag = (power.sqrt() * n_inv * (6.2 + frac * 10.5)).sqrt();
            let level = mag.clamp(0.0, 1.0);
            if level >= self.bands[b] {
                self.bands[b] = self.bands[b] * 0.30 + level * 0.70;
            } else {
                self.bands[b] = self.bands[b] * 0.76 + level * 0.24;
            }
            if self.bands[b] >= self.peaks[b] {
                self.peaks[b] = self.bands[b];
                self.peak_vel[b] = 0.0;
            } else {
                self.peak_vel[b] += 0.014;
                self.peaks[b] = (self.peaks[b] - self.peak_vel[b]).max(self.bands[b]);
            }
        }
    }

    // =========================================================================
    // Informative 1990s × Charm UI Components
    // =========================================================================

    /// Draws the 1-row Lip-Gloss / Amiga Copper hairline divider (`Row 2`).
    /// Calm static gradient when idle; smooth pulse sweep ONLY while GPU is busy!
    pub(super) fn draw_copper_rule(&self, scr: &mut ScreenBuf, w: usize, y: usize, gpu_busy: bool) {
        let pal = self.palette();
        if y >= scr.dims().1 {
            return;
        }
        let phase_shift = if gpu_busy { -self.time_s * 0.85 } else { 0.0 };
        for x in 0..w {
            let u = (x as f32) / (w.max(1) as f32);
            let col = if gpu_busy {
                let wave = ((u * 2.5 + phase_shift * 2.2).sin() * 0.5 + 0.5).powi(2);
                let base = copper_loop(&pal, u + phase_shift);
                lerp_rgb(pal.border_dim, base, 0.45 + 0.55 * wave)
            } else {
                let base = copper_gradient(&pal, u);
                lerp_rgb(pal.border_dim, base, 0.52)
            };
            scr.put_char(x, y, '─', Style::new(col, pal.bg_void));
        }
    }

    /// Segmented GDDR6X VRAM bar (`█` Thinker, `▓` Aux/Spec/LoRA, `▒` Voice, `░` Free).
    pub(super) fn draw_vram_bar(
        &self,
        scr: &mut ScreenBuf,
        x: usize,
        y: usize,
        cells: usize,
        bg: Rgb,
        snap: &TelemetrySnapshot<'_>,
    ) -> usize {
        let pal = self.palette();
        let (tot_mb, th_mb, aux_mb, vc_mb) = if let Some(hw) = snap.hw {
            (hw.total_mb.max(1.0), hw.thinker_mb, hw.aux_mb, hw.voice_mb)
        } else {
            let est_th = snap.gguf_size_gb * 1024.0 + 400.0;
            let est_aux = if snap.is_inkling && snap.spec_on { 447.0 } else { 0.0 };
            (10_240.0, est_th, est_aux, 0.0)
        };
        let n = cells.max(6);
        let c_th = ((th_mb / tot_mb) * (n as f64)).round() as usize;
        let c_aux = if aux_mb > 20.0 {
            (((aux_mb / tot_mb) * (n as f64)).round() as usize).max(1)
        } else {
            0
        };
        let c_vc = if vc_mb > 20.0 {
            (((vc_mb / tot_mb) * (n as f64)).round() as usize).max(1)
        } else {
            0
        };

        let mut cx = x;
        for i in 0..n {
            let (ch, fg) = if i < c_th {
                ('█', pal.grad_a)
            } else if i < c_th + c_aux {
                ('▓', pal.grad_b)
            } else if i < c_th + c_aux + c_vc {
                ('▒', pal.grad_c)
            } else {
                ('░', pal.border_dim)
            };
            cx += scr.put_char(cx, y, ch, Style::new(fg, bg));
        }
        cx
    }

    /// Real measured `tok/s` sparkline across session turns (`0` fake noise!).
    pub(super) fn draw_tps_sparkline(
        &self,
        scr: &mut ScreenBuf,
        x: usize,
        y: usize,
        max_pts: usize,
        bg: Rgb,
        live_tps: f64,
    ) -> usize {
        const SPARK: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
        let pal = self.palette();
        let mut vals: Vec<f64> = self.tps_history.clone();
        if live_tps > 0.5 && (vals.is_empty() || (vals.last().copied().unwrap_or(0.0) - live_tps).abs() > 0.2) {
            vals.push(live_tps);
        }
        if vals.is_empty() {
            return x;
        }
        let start = vals.len().saturating_sub(max_pts);
        let slice = &vals[start..];
        let peak = slice.iter().copied().fold(140.0f64, f64::max);
        let mut cx = x;
        for &v in slice {
            let norm = (v / peak).clamp(0.08, 1.0);
            let idx = ((norm * 7.0).round() as usize).min(7);
            let col = copper_gradient(&pal, norm as f32);
            cx += scr.put_char(cx, y, SPARK[idx], Style::new(col, bg));
        }
        cx
    }

    /// Live 16-band Hann-Goertzel voice spectrum (`[····]` flat baseline when silent).
    pub(super) fn draw_voice_spectrum(
        &self,
        scr: &mut ScreenBuf,
        x: usize,
        y: usize,
        max_x: usize,
        bg: Rgb,
    ) -> usize {
        const BARS: [char; 9] = ['·', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
        let pal = self.palette();
        if x + NUM_BANDS + 2 >= max_x {
            return x;
        }
        let mut cx = scr.put_char(x, y, '[', Style::new(pal.border_mid, bg));
        for b in 0..NUM_BANDS {
            let lvl = self.bands[b].clamp(0.0, 1.0);
            let pk = self.peaks[b].clamp(0.0, 1.0);
            let idx = ((lvl * 8.0).round() as usize).min(8);
            let ch = if idx == 0 && pk > 0.16 { '˙' } else { BARS[idx] };
            let fg = if !self.audio_active && idx == 0 {
                pal.border_dim
            } else if lvl > 0.78 {
                FG_CORAL
            } else if lvl > 0.48 {
                FG_AMBER
            } else if lvl > 0.08 {
                FG_GREEN
            } else {
                pal.border_mid
            };
            cx += scr.put_char(cx, y, ch, Style::new(fg, bg));
        }
        cx += scr.put_char(cx, y, ']', Style::new(pal.border_mid, bg));
        cx
    }

    /// Calm, high-signal Empty-Chat Session Brief (Charm × 1990s Copper Wordmark + Live Hardware Readiness).
    pub(super) fn draw_session_brief(
        &self,
        scr: &mut ScreenBuf,
        x0: usize,
        y0: usize,
        w: usize,
        h: usize,
        snap: &TelemetrySnapshot<'_>,
    ) {
        let pal = self.palette();
        if w < 44 || h < 7 {
            return;
        }

        let card_w = w.saturating_sub(6).min(86);
        let cx0 = x0 + (w.saturating_sub(card_w)) / 2;
        let card_h = 15usize.min(h.saturating_sub(1));
        let cy0 = y0 + (h.saturating_sub(card_h)) / 3;

        // Subtle 1990s × Charm Copper-Gradient Wordmark above/inside the session card
        const LOGO: [&str; 3] = [
            "█▀█ ▀█▀ ▀▄▀   ╱╱   █▀▀ ▄▀█ ▄█ █▀█ ▀█",
            "█▀▄  █  █ █  ╱╱    █▄█ █▀█  █ █▄█ █▄",
            "▀▀▀  ▀  ▀ ▀        ▀▀▀ ▀ ▀  ▀ ▀▀▀ ▀▀",
        ];
        let mut row = cy0;
        if card_h >= 13 {
            for (li, ln) in LOGO.iter().enumerate() {
                if row >= y0 + h {
                    break;
                }
                for (ci, ch) in ln.chars().enumerate() {
                    let u = (ci as f32) / (ln.chars().count().max(1) as f32) + (li as f32) * 0.12;
                    let col = copper_gradient(&pal, u);
                    scr.put_char(cx0 + 2 + ci, row, ch, Style::new(col, pal.bg_void).bold());
                }
                row += 1;
            }
            if row < y0 + h {
                scr.put_str(
                    cx0 + 2,
                    row,
                    cx0 + card_w,
                    "Native macOS ARM64 Rust · 100% NVIDIA RTX 3080 (sm_86) · Zero Docker · Zero Crates",
                    Style::new(pal.text_muted, pal.bg_void),
                );
                row += 2;
            }
        }

        // Rounded Telemetry Readiness Card (`╭─ SESSION READY ─╮`)
        if row + 4 >= y0 + h {
            return;
        }
        let inner_w = card_w.saturating_sub(2);
        scr.fill_row_range_help(cx0, row, card_w, pal.bg_surface);
        scr.put_char(cx0, row, '╭', Style::new(pal.border_mid, pal.bg_surface));
        let hdr = "─ SYSTEM & MODEL TELEMETRY ";
        let mut tx = scr.put_str(cx0 + 1, row, cx0 + card_w - 1, hdr, Style::new(pal.grad_b, pal.bg_surface).bold());
        while tx < cx0 + card_w - 1 {
            tx += scr.put_char(tx, row, '─', Style::new(pal.border_dim, pal.bg_surface));
        }
        scr.put_char(cx0 + card_w - 1, row, '╮', Style::new(pal.border_mid, pal.bg_surface));
        row += 1;

        let (used_gib, free_gib, aux_desc) = if let Some(hw) = snap.hw {
            (
                (hw.thinker_mb + hw.aux_mb + hw.voice_mb) / 1024.0,
                hw.free_mb / 1024.0,
                hw.aux_label.as_str(),
            )
        } else {
            (snap.gguf_size_gb, (10.0 - snap.gguf_size_gb - 0.5).max(0.0), "ready")
        };

        let arch_summary = if snap.is_inkling {
            format!(
                "{} · 24→48L 128×8 Sparse MoE ({:.2} GB/tok active)",
                snap.model_name, snap.active_gb_per_tok
            )
        } else {
            format!("{} · {:.2} GB GGUF (read-only mmap)", snap.model_name, snap.gguf_size_gb)
        };
        let mode_summary = if snap.is_inkling {
            if snap.spec_on {
                format!("DSpark Block-8 Speculative (γ={}) · Effort {}", snap.gamma, snap.effort)
            } else {
                format!("48L Persistent B=1 Megakernel (~122 tok/s) · Effort {}", snap.effort)
            }
        } else if snap.spec_on {
            "BM=4 Lossless Speculative Megakernel".to_string()
        } else {
            "68-SM Persistent T=1 Megakernel".to_string()
        };
        let link_str = if snap.daemon_online {
            format!("{} SMs · {} · {:.1}ms IPC", snap.active_sms, aux_desc, snap.ping_ms)
        } else {
            "daemon offline".to_string()
        };
        let vram_summary = format!(
            "{:.2} GiB allocated · {:.2} GiB free high VRAM ({})",
            used_gib, free_gib, link_str
        );
        let voice_summary = format!(
            "Qwen3-ASR-{} + Qwen3-TTS-{} ({} · 24 kHz Code2Wav)",
            snap.voice_model_ref.to_uppercase(),
            snap.voice_model_ref.to_uppercase(),
            snap.voice_speaker
        );

        let rows: [(&str, String, Rgb); 4] = [
            ("Backbone ", arch_summary, pal.text_hi),
            ("Compute  ", mode_summary, pal.grad_c),
            ("VRAM     ", vram_summary, pal.grad_b),
            ("Voice    ", voice_summary, pal.text_muted),
        ];

        for (lbl, val, col) in &rows {
            if row + 1 >= y0 + h {
                break;
            }
            scr.fill_row_range_help(cx0, row, card_w, pal.bg_surface);
            scr.put_char(cx0, row, '│', Style::new(pal.border_dim, pal.bg_surface));
            let lx = scr.put_str(cx0 + 2, row, cx0 + inner_w, lbl, Style::new(pal.text_dim, pal.bg_surface));
            scr.put_str(lx + 1, row, cx0 + card_w - 2, val, Style::new(*col, pal.bg_surface));
            scr.put_char(cx0 + card_w - 1, row, '│', Style::new(pal.border_dim, pal.bg_surface));
            row += 1;
        }

        if row < y0 + h {
            scr.fill_row_range_help(cx0, row, card_w, pal.bg_surface);
            scr.put_char(cx0, row, '╰', Style::new(pal.border_mid, pal.bg_surface));
            for x in (cx0 + 1)..(cx0 + card_w - 1) {
                scr.put_char(x, row, '─', Style::new(pal.border_dim, pal.bg_surface));
            }
            scr.put_char(cx0 + card_w - 1, row, '╯', Style::new(pal.border_mid, pal.bg_surface));
            row += 2;
        }

        if row < y0 + h {
            let hints = "  / commands · Ctrl+R speak · /talk hands-free · Ctrl+O toggle <think> · Ctrl+B sidebar · Ctrl+T theme";
            scr.put_str(cx0, row, cx0 + card_w + 4, hints, Style::new(pal.text_dim, pal.bg_void));
        }
    }

    /// Right-Side Hardware & Session Telemetry Inspector (`34` columns, `Ctrl+B` or `/inspector`).
    pub(super) fn draw_inspector_sidebar(
        &self,
        scr: &mut ScreenBuf,
        x0: usize,
        y0: usize,
        w: usize,
        h: usize,
        snap: &TelemetrySnapshot<'_>,
    ) {
        let pal = self.palette();
        if w < 26 || h < 8 {
            return;
        }
        let max_x = x0 + w - 1;
        for r in 0..h {
            let y = y0 + r;
            scr.fill_row_range_help(x0, y, w, pal.bg_header);
            scr.put_char(x0, y, '│', Style::new(pal.border_dim, pal.bg_header));
        }

        let lx = x0 + 2;
        let bar_w = w.saturating_sub(4).min(26);
        let mut y = y0;

        let put_sec = |scr: &mut ScreenBuf, y: usize, title: &str| {
            let sx = scr.put_str(lx, y, max_x, title, Style::new(pal.grad_b, pal.bg_header).bold());
            for x in (sx + 1)..max_x {
                scr.put_char(x, y, '─', Style::new(pal.border_dim, pal.bg_header));
            }
        };

        // 1. GDDR6X VRAM Allocation Map
        put_sec(scr, y, "VRAM MAP");
        y += 1;
        if y < y0 + h {
            self.draw_vram_bar(scr, lx, y, bar_w, pal.bg_header, snap);
            y += 1;
        }
        let (th_g, aux_g, aux_lbl, vc_g, free_g) = if let Some(hw) = snap.hw {
            (
                hw.thinker_mb / 1024.0,
                hw.aux_mb / 1024.0,
                hw.aux_label.as_str(),
                hw.voice_mb / 1024.0,
                hw.free_mb / 1024.0,
            )
        } else {
            (
                snap.gguf_size_gb,
                if snap.is_inkling && snap.spec_on { 0.44 } else { 0.0 },
                if snap.is_inkling && snap.spec_on { "DSpark" } else { "standby" },
                0.0,
                (10.0 - snap.gguf_size_gb - 0.5).max(0.0),
            )
        };
        let vram_rows: [(char, Rgb, &str, String); 4] = [
            ('█', pal.grad_a, "Thinker+KV", format!("{:>5.2}G", th_g)),
            ('▓', pal.grad_b, aux_lbl, format!("{:>5.2}G", aux_g)),
            ('▒', pal.grad_c, "Voice Arena", format!("{:>5.2}G", vc_g)),
            ('░', pal.text_dim, "Free High", format!("{:>5.2}G", free_g)),
        ];
        for (dot, col, name, val) in &vram_rows {
            if y >= y0 + h {
                return;
            }
            scr.put_char(lx, y, *dot, Style::new(*col, pal.bg_header));
            scr.put_str(lx + 2, y, max_x - 7, name, Style::new(pal.text_muted, pal.bg_header));
            scr.put_str(max_x.saturating_sub(6), y, max_x, val, Style::new(pal.text_hi, pal.bg_header));
            y += 1;
        }

        // 2. Compute Roofline & Sparkline
        y += 1;
        if y + 3 >= y0 + h {
            return;
        }
        put_sec(scr, y, "ROOFLINE");
        y += 1;
        let eff_bw = snap.active_tps * snap.active_gb_per_tok;
        let roof_pct = ((eff_bw / GDDR6X_PEAK_GBS) * 100.0).clamp(0.0, 100.0);
        let ms_tok = if snap.active_tps > 0.1 { 1000.0 / snap.active_tps } else { 0.0 };
        let r_rows: [(&str, String, Rgb); 4] = [
            (
                "Decode",
                if snap.active_tps > 0.0 {
                    format!("{:.1} tok/s ({:.1}ms)", snap.active_tps, ms_tok)
                } else if snap.is_streaming {
                    "Prefilling...".into()
                } else {
                    "—".into()
                },
                pal.grad_c,
            ),
            (
                "Bus BW",
                if eff_bw > 0.0 {
                    format!("{:.0} GB/s ({:.0}%)", eff_bw, roof_pct)
                } else {
                    format!("0 / {:.0} GB/s", GDDR6X_PEAK_GBS)
                },
                pal.grad_b,
            ),
            (
                "Prefill",
                if snap.last_ttft_ms > 0.0 {
                    if snap.last_prefill_tps > 1.0 {
                        format!("{:.0} tok/s · {:.0}ms", snap.last_prefill_tps, snap.last_ttft_ms)
                    } else {
                        format!("TTFT {:.0}ms · {} QMD", snap.last_ttft_ms, snap.last_launches.max(1))
                    }
                } else {
                    "—".into()
                },
                pal.text_hi,
            ),
            (
                "Mode",
                if snap.is_inkling {
                    if snap.spec_on {
                        format!("DSpark γ={} (+{})", snap.gamma, snap.last_spec_extra)
                    } else {
                        format!("48L B=1 · {} QMD", snap.last_launches.max(2))
                    }
                } else if snap.spec_on {
                    format!("Spec BM=4 (+{})", snap.last_spec_extra)
                } else {
                    format!("T=1 · {} QMD", snap.last_launches.max(2))
                },
                pal.text_muted,
            ),
        ];
        for (k, v, c) in &r_rows {
            if y >= y0 + h {
                return;
            }
            scr.put_str(lx, y, lx + 9, k, Style::new(pal.text_dim, pal.bg_header));
            scr.put_str(lx + 9, y, max_x, v, Style::new(*c, pal.bg_header));
            y += 1;
        }
        if y < y0 + h && (!self.tps_history.is_empty() || snap.active_tps > 0.5) {
            scr.put_str(lx, y, lx + 9, "Trend", Style::new(pal.text_dim, pal.bg_header));
            self.draw_tps_sparkline(scr, lx + 9, y, bar_w.saturating_sub(9), pal.bg_header, snap.active_tps);
            y += 1;
        }

        // 3. Context & Prefix Cache
        y += 1;
        if y + 3 >= y0 + h {
            return;
        }
        put_sec(scr, y, "CONTEXT & THINK");
        y += 1;
        let ctx_pct = ((snap.ctx_used as f64 / snap.max_ctx.max(1) as f64) * 100.0).min(100.0);
        let hit_pct = if snap.ctx_used > 0 {
            ((snap.cached_prefix as f64 / snap.ctx_used as f64) * 100.0).min(100.0)
        } else {
            0.0
        };
        let c_rows: [(&str, String); 4] = [
            ("Window", format!("{}/{} ({:.0}%)", snap.ctx_used, snap.max_ctx, ctx_pct)),
            ("Cache", format!("{} hit ({:.0}%)", snap.cached_prefix, hit_pct)),
            (
                "Output",
                format!(
                    "{}w think · {}w ans",
                    snap.think_words, snap.answer_words
                ),
            ),
            ("Sample", format!("T={:.2} · max {}", snap.temp, snap.max_tokens)),
        ];
        for (k, v) in &c_rows {
            if y >= y0 + h {
                return;
            }
            scr.put_str(lx, y, lx + 9, k, Style::new(pal.text_dim, pal.bg_header));
            scr.put_str(lx + 9, y, max_x, v, Style::new(pal.text_hi, pal.bg_header));
            y += 1;
        }

        // 4. Co-Resident Voice & Live Audio Spectrum
        y += 1;
        if y + 3 >= y0 + h {
            return;
        }
        put_sec(scr, y, "VOICE I/O");
        y += 1;
        let asr_mb = snap.hw.map(|h| h.asr_mb).unwrap_or(0.0);
        let tts_mb = snap.hw.map(|h| h.tts_mb).unwrap_or(0.0);
        let stt_val = if snap.is_recording {
            "Recording 16kHz...".to_string()
        } else if snap.is_transcribing {
            "GPU AuT+Decode...".to_string()
        } else if snap.last_asr_ms > 0.0 {
            format!("ASR-{} ({:.0}ms)", snap.voice_model_ref, snap.last_asr_ms)
        } else if asr_mb > 1.0 {
            format!("ASR-{} ({:.0}M warm)", snap.voice_model_ref, asr_mb)
        } else {
            format!("ASR-{} standby", snap.voice_model_ref)
        };
        let tts_val = if snap.is_speaking {
            format!("Speaking ({} · {:.0}ms)", snap.voice_speaker, snap.last_ttfa_ms)
        } else if tts_mb > 1.0 {
            format!("{} ({:.0}M warm)", snap.voice_speaker, tts_mb)
        } else if snap.voice_enabled || snap.hands_free {
            format!("{} (ON)", snap.voice_speaker)
        } else {
            format!("{} (Ctrl+R)", snap.voice_speaker)
        };
        scr.put_str(lx, y, lx + 9, "STT", Style::new(pal.text_dim, pal.bg_header));
        scr.put_str(
            lx + 9,
            y,
            max_x,
            &stt_val,
            Style::new(if snap.is_recording { FG_GREEN } else { pal.text_hi }, pal.bg_header),
        );
        y += 1;
        if y < y0 + h {
            scr.put_str(lx, y, lx + 9, "TTS", Style::new(pal.text_dim, pal.bg_header));
            scr.put_str(
                lx + 9,
                y,
                max_x,
                &tts_val,
                Style::new(if snap.is_speaking { FG_AMBER } else { pal.text_muted }, pal.bg_header),
            );
            y += 1;
        }
        if y < y0 + h {
            self.draw_voice_spectrum(scr, lx, y, max_x, pal.bg_header);
        }
    }

    /// Floating Slash-Command Autocomplete Palette above the composer (Hermes / Claude / Pi style).
    pub(super) fn draw_slash_palette(
        &self,
        scr: &mut ScreenBuf,
        x0: usize,
        comp_top_y: usize,
        w: usize,
        items: &[SlashItem],
        selected: usize,
    ) {
        if items.is_empty() || comp_top_y < 4 || w < 36 {
            return;
        }
        let pal = self.palette();
        let max_vis = items.len().min(6).min(comp_top_y.saturating_sub(3));
        if max_vis == 0 {
            return;
        }
        let start = if selected >= max_vis {
            selected + 1 - max_vis
        } else {
            0
        };
        let box_h = max_vis + 2;
        let top_y = comp_top_y - box_h;
        let box_w = w.min(82);

        scr.fill_row_range_help(x0, top_y, box_w, pal.bg_surface);
        scr.put_char(x0, top_y, '╭', Style::new(pal.border_hi, pal.bg_surface));
        let hdr = "─ Commands (↑↓ select · Tab complete · Enter run) ";
        let mut hx = scr.put_str(x0 + 1, top_y, x0 + box_w - 1, hdr, Style::new(pal.grad_b, pal.bg_surface).bold());
        while hx < x0 + box_w - 1 {
            hx += scr.put_char(hx, top_y, '─', Style::new(pal.border_dim, pal.bg_surface));
        }
        scr.put_char(x0 + box_w - 1, top_y, '╮', Style::new(pal.border_hi, pal.bg_surface));

        for i in 0..max_vis {
            let idx = start + i;
            let it = &items[idx];
            let ry = top_y + 1 + i;
            let is_sel = idx == selected;
            let row_bg = if is_sel { pal.bg_header } else { pal.bg_surface };
            scr.fill_row_range_help(x0, ry, box_w, row_bg);
            scr.put_char(x0, ry, '│', Style::new(pal.border_hi, row_bg));
            let marker = if is_sel { "▸ " } else { "  " };
            let cmd_fg = if is_sel { pal.grad_c } else { pal.grad_b };
            let mut cx = scr.put_str(x0 + 1, ry, x0 + box_w - 2, marker, Style::new(pal.grad_c, row_bg).bold());
            cx = scr.put_str(cx, ry, x0 + 26, it.cmd, Style::new(cmd_fg, row_bg).bold());
            if !it.args.is_empty() {
                cx = scr.put_str(cx + 1, ry, x0 + 29, it.args, Style::new(pal.text_dim, row_bg));
            }
            let desc_x = (x0 + 28).max(cx + 2);
            let cur_tag = if it.current.is_empty() {
                String::new()
            } else {
                format!("[{}]", it.current)
            };
            let right_limit = (x0 + box_w - 2).saturating_sub(cur_tag.chars().count() + 1);
            scr.put_str(desc_x, ry, right_limit, it.desc, Style::new(pal.text_muted, row_bg));
            if !cur_tag.is_empty() {
                scr.put_str(right_limit + 1, ry, x0 + box_w - 1, &cur_tag, Style::new(FG_CYAN, row_bg));
            }
            scr.put_char(x0 + box_w - 1, ry, '│', Style::new(pal.border_hi, row_bg));
        }

        let bot_y = top_y + box_h - 1;
        scr.fill_row_range_help(x0, bot_y, box_w, pal.bg_surface);
        scr.put_char(x0, bot_y, '╰', Style::new(pal.border_hi, pal.bg_surface));
        for x in (x0 + 1)..(x0 + box_w - 1) {
            scr.put_char(x, bot_y, '─', Style::new(pal.border_dim, pal.bg_surface));
        }
        scr.put_char(x0 + box_w - 1, bot_y, '╯', Style::new(pal.border_hi, pal.bg_surface));
    }

    // =========================================================================
    // Opt-In `/demo` Fullscreen 60 FPS 1990s Megademo Showcase
    // =========================================================================

    pub(super) fn draw_megademo(&mut self, scr: &mut ScreenBuf, w: usize, h: usize, model_name: &str, active_sms: u32) {
        let pal = self.palette();
        scr.resize_and_clear(w, h, pal.bg_void);
        if w < 24 || h < 10 {
            return;
        }
        let scene_names = [
            "1/4 · PLASMA & AMIGA COPPERBARS",
            "2/4 · 3D GLENZ VECTOR & STARFIELD",
            "3/4 · DOOM FIRE & KEFRENS RASTERS",
            "4/4 · DYCP SINE SCROLLER & HORIZON",
        ];
        scr.fill_row(0, pal.bg_header);
        let mut x = scr.put_str(
            0,
            0,
            w,
            &format!(" RTX DEMO // {} ", scene_names[self.showcase_scene % 4]),
            Style::new(pal.bg_void, pal.grad_b).bold(),
        );
        let mod_lbl = if self.mod_playing.load(Ordering::Relaxed) {
            " ♫ MOD:ON "
        } else {
            " M:Synth "
        };
        x = scr.put_str(x + 1, 0, w, mod_lbl, Style::new(pal.grad_c, pal.bg_header).bold());
        let rhs = "1-4/Space:Scene · T:Theme · M:Chiptune · Esc:Return ";
        let rx = w.saturating_sub(rhs.chars().count());
        if rx > x + 1 {
            scr.put_str(rx, 0, w, rhs, Style::new(pal.text_muted, pal.bg_header));
        }

        let canvas_rows = h.saturating_sub(1);
        let pw = w;
        let ph = canvas_rows * 2;
        self.pix_w = pw;
        self.pix_h = ph;
        self.pixels.clear();
        self.pixels.resize(pw * ph, pal.bg_void);

        match self.showcase_scene % 4 {
            0 => self.scene_plasma(&pal),
            1 => self.scene_vector_stars(&pal),
            2 => self.scene_fire_kefrens(&pal),
            _ => self.scene_dycp_horizon(&pal, model_name, active_sms),
        }

        for r in 0..canvas_rows {
            for c in 0..w {
                let top = self.pixels[(r * 2) * pw + c];
                let bot = self.pixels[(r * 2 + 1) * pw + c];
                if top == bot {
                    scr.put_char(c, 1 + r, ' ', Style::new(top, top));
                } else {
                    scr.put_char(c, 1 + r, '▀', Style::new(top, bot));
                }
            }
        }
    }

    #[inline]
    fn put_px(&mut self, x: isize, y: isize, col: Rgb) {
        if x >= 0 && y >= 0 && (x as usize) < self.pix_w && (y as usize) < self.pix_h {
            self.pixels[(y as usize) * self.pix_w + (x as usize)] = col;
        }
    }

    fn draw_line_px(&mut self, mut x0: isize, mut y0: isize, x1: isize, y1: isize, col: Rgb) {
        let dx = (x1 - x0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let dy = -(y1 - y0).abs();
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;
        for _ in 0..1024 {
            self.put_px(x0, y0, col);
            if x0 == x1 && y0 == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x0 += sx;
            }
            if e2 <= dx {
                err += dx;
                y0 += sy;
            }
        }
    }

    fn render_starfield_px(&mut self, pal: &Palette) {
        let (pw, ph) = (self.pix_w as f32, self.pix_h as f32);
        let (cx, cy) = (pw * 0.5, ph * 0.5);
        for i in 0..NUM_STARS {
            let st = self.stars[i];
            let inv_z = 1.0 / st.z.max(0.06);
            let sx = (cx + st.x * inv_z * (pw * 0.36)) as isize;
            let sy = (cy + st.y * inv_z * (ph * 0.42)) as isize;
            let bright = ((2.1 - st.z) / 2.0).clamp(0.15, 1.0);
            self.put_px(sx, sy, lerp_rgb(pal.border_dim, Rgb(245, 250, 255), bright));
        }
    }

    fn scene_plasma(&mut self, pal: &Palette) {
        let (pw, ph, t) = (self.pix_w, self.pix_h, self.time_s);
        let mut bars = [(0.0f32, Rgb(0, 0, 0)); 5];
        for i in 0..5 {
            let fi = i as f32;
            let yc = 0.50 + 0.36 * (t * (1.25 + fi * 0.21) + fi * 1.15).sin();
            bars[i] = (yc * (ph as f32), copper_loop(pal, fi * 0.22 - t * 0.15));
        }
        for py in 0..ph {
            let yf = (py as f32) * 0.065;
            for px in 0..pw {
                let xf = (px as f32) * 0.045;
                let u = ((xf + t * 1.6).sin() + (yf - t * 1.3).sin() + ((xf + yf) * 0.7 + t).sin()) * 0.16 + 0.5;
                let mut col = scale_rgb(copper_loop(pal, u), 0.52);
                for &(by, bc) in &bars {
                    let dist = ((py as f32) - by).abs();
                    if dist < 4.0 {
                        let k = 1.0 - dist / 4.0;
                        let spec = if dist < 1.1 { lerp_rgb(bc, Rgb(255, 255, 255), 0.75) } else { scale_rgb(bc, k * k) };
                        col = lerp_rgb(col, spec, k);
                    }
                }
                self.pixels[py * pw + px] = col;
            }
        }
    }

    fn scene_vector_stars(&mut self, pal: &Palette) {
        self.render_starfield_px(pal);
        const VERTS: [(f32, f32, f32); 14] = [
            (-1.0, -1.0, -1.0), (1.0, -1.0, -1.0), (1.0, 1.0, -1.0), (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),  (1.0, -1.0, 1.0),  (1.0, 1.0, 1.0),  (-1.0, 1.0, 1.0),
            (0.0, 0.0, -1.75),  (0.0, 0.0, 1.75),  (-1.75, 0.0, 0.0), (1.75, 0.0, 0.0),
            (0.0, -1.75, 0.0),  (0.0, 1.75, 0.0),
        ];
        const EDGES: [(usize, usize); 24] = [
            (0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7), (8, 0), (8, 1), (8, 2), (8, 3),
            (9, 4), (9, 5), (9, 6), (9, 7), (10, 11), (12, 13), (8, 13), (9, 12),
        ];
        let (pw, ph, t) = (self.pix_w as f32, self.pix_h as f32, self.time_s);
        let (cx, cy, scale) = (pw * 0.5, ph * 0.5, ph.min(pw * 0.65) * 1.35);
        let (sx, cx_r) = (t * 0.92).sin_cos();
        let (sy, cy_r) = (t * 1.27).sin_cos();
        let (sz, cz_r) = (t * 0.61).sin_cos();
        let mut proj = [(0isize, 0isize, 0.0f32); 14];
        for (i, &(vx, vy, vz)) in VERTS.iter().enumerate() {
            let (y1, z1) = (vy * cx_r - vz * sx, vy * sx + vz * cx_r);
            let (x2, z2) = (vx * cy_r + z1 * sy, -vx * sy + z1 * cy_r);
            let (x3, y3) = (x2 * cz_r - y1 * sz, x2 * sz + y1 * cz_r);
            let p = scale / (4.2 + z2).max(1.2);
            proj[i] = ((cx + x3 * p * 1.15).round() as isize, (cy + y3 * p).round() as isize, z2);
        }
        for (ei, &(a, b)) in EDGES.iter().enumerate() {
            let ((x0, y0, za), (x1, y1, zb)) = (proj[a], proj[b]);
            let depth = ((2.0 - (za + zb) * 0.5) / 3.6).clamp(0.20, 1.0);
            let col = lerp_rgb(pal.border_dim, if ei >= 12 { pal.grad_c } else { pal.grad_b }, depth);
            self.draw_line_px(x0, y0, x1, y1, col);
        }
    }

    fn scene_fire_kefrens(&mut self, pal: &Palette) {
        let (pw, ph) = (self.pix_w, self.pix_h);
        if self.fire_w != pw || self.fire_h != ph || self.fire_buf.len() != pw * ph {
            self.fire_w = pw;
            self.fire_h = ph;
            self.fire_buf.resize(pw * ph, 0);
        }
        let bass = (self.bands[0] + self.bands[1] + self.bands[2]) * 0.33;
        if ph >= 2 {
            for x in 0..pw {
                self.fire_buf[(ph - 1) * pw + x] = if self.next_rand_f32() < 0.82 + bass * 0.15 { 255 } else { 140 };
            }
        }
        for y in 0..ph.saturating_sub(1) {
            for x in 0..pw {
                let drift = ((self.next_rand_f32() * 3.0) as usize).saturating_sub(1);
                let below = self.fire_buf[(y + 1) * pw + (x + drift).min(pw - 1)] as i32;
                let decay = ((self.next_rand_f32() * (280.0 / (ph.max(16) as f32)).max(4.5)) as i32).max(1);
                self.fire_buf[y * pw + x] = (below - decay).max(0) as u8;
            }
        }
        for y in 0..ph {
            for x in 0..pw {
                let hval = self.fire_buf[y * pw + x] as f32 / 255.0;
                let col = if hval < 0.25 {
                    lerp_rgb(pal.bg_void, Rgb(120, 12, 18), hval * 4.0)
                } else if hval < 0.55 {
                    lerp_rgb(Rgb(120, 12, 18), Rgb(245, 96, 18), (hval - 0.25) / 0.30)
                } else if hval < 0.82 {
                    lerp_rgb(Rgb(245, 96, 18), Rgb(255, 216, 56), (hval - 0.55) / 0.27)
                } else {
                    lerp_rgb(Rgb(255, 216, 56), Rgb(255, 255, 245), (hval - 0.82) / 0.18)
                };
                self.pixels[y * pw + x] = col;
            }
        }
    }

    fn scene_dycp_horizon(&mut self, pal: &Palette, model_name: &str, active_sms: u32) {
        self.render_starfield_px(pal);
        let (pw, ph) = (self.pix_w, self.pix_h);
        let horizon = ph * 54 / 100;
        for py in horizon..ph {
            let dy = ((py - horizon + 1) as f32) / ((ph - horizon).max(1) as f32);
            let z = 1.0 / dy.max(0.04);
            for px in 0..pw {
                let nx = ((px as f32) / (pw.max(1) as f32) - 0.5) * 2.0;
                let chk = (((nx * z * 2.6).floor() as i32) ^ (((z * 1.4 + self.time_s * 4.5).floor()) as i32)) & 1;
                let tile = if chk == 0 { pal.grad_b } else { pal.bg_header };
                self.pixels[py * pw + px] = lerp_rgb(pal.bg_void, tile, dy.powf(0.75) * 0.72);
            }
        }
        let msg = format!("   RTX 3080 GA102 · {} SMs · {}   ", active_sms, model_name.to_ascii_uppercase());
        let scale = if ph >= 36 { 2isize } else { 1isize };
        let total_px = (msg.chars().count() as isize) * 6 * scale;
        let scroll_x = -(((self.time_s * 40.0 * (scale as f32)) as isize).rem_euclid(total_px.max(1)));
        let base_y = (horizon as isize / 2 - 3 * scale).max(2);
        for off in [scroll_x, scroll_x + total_px] {
            let mut pen_x = off;
            for (ci, ch) in msg.chars().enumerate() {
                let bits = glyph_5x5(ch);
                for gx in 0..5isize {
                    let col_x = pen_x + gx * scale;
                    let wy = (((col_x as f32) * 0.085 + self.time_s * 4.4).sin() * 4.0 * (scale as f32)) as isize;
                    for gy in 0..5isize {
                        if ((bits >> ((4 - gy) * 5 + (4 - gx))) & 1) != 0 {
                            let col = copper_loop(pal, (ci as f32) * 0.1 + (gy as f32) * 0.14 - self.time_s * 0.5);
                            for sy in 0..scale {
                                for sx in 0..scale {
                                    self.put_px(col_x + sx, base_y + wy + gy * scale + sy, col);
                                }
                            }
                        }
                    }
                }
                pen_x += 6 * scale;
            }
        }
    }
}

fn glyph_5x5(ch: char) -> u32 {
    match ch.to_ascii_uppercase() {
        'A' => 0b01110_10001_11111_10001_10001,
        'B' => 0b11110_10001_11110_10001_11110,
        'C' => 0b01111_10000_10000_10000_01111,
        'D' => 0b11110_10001_10001_10001_11110,
        'E' => 0b11111_10000_11110_10000_11111,
        'F' => 0b11111_10000_11110_10000_10000,
        'G' => 0b01111_10000_10011_10001_01111,
        'H' => 0b10001_10001_11111_10001_10001,
        'I' => 0b11111_00100_00100_00100_11111,
        'K' => 0b10010_10100_11000_10100_10010,
        'L' => 0b10000_10000_10000_10000_11111,
        'M' => 0b10001_11011_10101_10001_10001,
        'N' => 0b10001_11001_10101_10011_10001,
        'O' | '0' => 0b01110_10001_10001_10001_01110,
        'P' => 0b11110_10001_11110_10000_10000,
        'Q' => 0b01110_10001_10101_10010_01101,
        'R' => 0b11110_10001_11110_10100_10010,
        'S' | '5' => 0b01111_10000_01110_00001_11110,
        'T' => 0b11111_00100_00100_00100_00100,
        'U' => 0b10001_10001_10001_10001_01110,
        'V' => 0b10001_10001_10001_01010_00100,
        'W' => 0b10001_10001_10101_11011_10001,
        'X' => 0b10001_01010_00100_01010_10001,
        'Y' => 0b10001_01010_00100_00100_00100,
        'Z' | '2' => 0b11111_00010_00100_01000_11111,
        '1' => 0b00100_01100_00100_00100_01110,
        '3' => 0b11110_00001_01110_00001_11110,
        '4' => 0b10010_10010_11111_00010_00010,
        '6' => 0b01111_10000_11110_10001_01110,
        '7' => 0b11111_00010_00100_01000_01000,
        '8' => 0b01110_10001_01110_10001_01110,
        '9' => 0b01110_10001_01111_00001_11110,
        '-' => 0b00000_00000_11111_00000_00000,
        '·' | '•' => 0b00000_00100_01110_00100_00000,
        _ => 0,
    }
}

#[inline]
fn midi_to_hz(note: i32) -> f32 {
    440.0 * 2.0f32.powf((note - 69) as f32 / 12.0)
}

fn run_tracker_chiptune_loop(
    player: &mut CoreAudioPlayer,
    cancel: &AtomicBool,
    viz_pcm: &Arc<Mutex<Vec<f32>>>,
) {
    const TICK_SAMPLES: usize = 480;
    const TICKS_PER_ROW: usize = 6;
    const CHORDS: [[i32; 3]; 8] = [
        [60, 63, 67], [60, 63, 67], [56, 60, 63], [63, 67, 70],
        [58, 62, 65], [53, 56, 60], [55, 59, 62], [55, 62, 65],
    ];
    const BASS_ROOTS: [i32; 8] = [36, 36, 32, 39, 34, 29, 31, 31];
    const LEAD: [i32; 64] = [
        72, 0, 75, 74, 72, 0, 67, 0,  72, 74, 75, 79, 77, 75, 74, 72,
        68, 0, 72, 0,  75, 0, 79, 77, 75, 0,  74, 0,  70, 0,  67, 0,
        70, 0, 74, 75, 77, 0, 74, 0,  65, 0,  68, 72, 75, 74, 72, 68,
        67, 0, 71, 74, 79, 0, 77, 74, 75, 74, 72, 71, 72, 0,  67, 0,
    ];

    let sr_inv = 1.0 / (TTS_SAMPLE_RATE as f32);
    let (mut ph_arp, mut ph_bass, mut ph_lead, mut ph_kick) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let mut lfsr = 0xACE1u32;
    let mut delay_ring = vec![0.0f32; 5760];
    let (mut delay_idx, mut row, mut tick) = (0usize, 0usize, 0usize);
    let mut lead_note = 72i32;
    let (mut lead_env, mut bass_env, mut kick_env, mut snare_env, mut hat_env) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let mut chunk = [0.0f32; TICK_SAMPLES];

    while !cancel.load(Ordering::Relaxed) {
        let bar = (row / 8) % 8;
        let step_in_bar = row % 8;
        let arp_semi = CHORDS[bar][tick % 3];
        if tick == 0 {
            bass_env = 1.0;
            if LEAD[row % 64] > 0 {
                lead_note = LEAD[row % 64];
                lead_env = 1.0;
            }
            if matches!(step_in_bar, 0 | 3 | 6) {
                kick_env = 1.0;
                ph_kick = 0.0;
            }
            if matches!(step_in_bar, 2 | 6) && row % 4 == 2 {
                snare_env = 1.0;
            }
            hat_env = if step_in_bar % 2 == 1 { 0.55 } else { 0.25 };
        }
        let f_arp = midi_to_hz(arp_semi);
        let f_bass = midi_to_hz(BASS_ROOTS[bar] + if step_in_bar % 2 == 1 { 12 } else { 0 });
        let f_lead = midi_to_hz(lead_note);
        let pwm = 0.30 + 0.22 * ((row as f32) * 0.18).sin();

        for s in chunk.iter_mut() {
            ph_arp = (ph_arp + f_arp * sr_inv).fract();
            let c_arp = if ph_arp < pwm { 0.14 } else { -0.14 };

            ph_bass = (ph_bass + f_bass * sr_inv).fract();
            let c_bass = ((ph_bass * 2.0 - 1.0) * 0.6 + if ph_bass < 0.5 { 0.4 } else { -0.4 }) * bass_env * 0.24;
            bass_env *= 0.99935;

            let vib = 1.0 + 0.0045 * (ph_arp * TAU).sin();
            ph_lead = (ph_lead + f_lead * vib * sr_inv).fract();
            let dry_lead = (if ph_lead < 0.42 { 1.0 } else { -1.0 }) * lead_env * 0.16;
            lead_env *= 0.99965;
            let echoed = dry_lead + delay_ring[delay_idx] * 0.38;
            delay_ring[delay_idx] = echoed;
            delay_idx = (delay_idx + 1) % delay_ring.len();

            ph_kick = (ph_kick + (44.0 + 135.0 * kick_env * kick_env) * sr_inv).fract();
            let c_kick = (ph_kick * TAU).sin() * kick_env * 0.30;
            kick_env *= 0.9982;

            let bit = (lfsr ^ (lfsr >> 2) ^ (lfsr >> 3) ^ (lfsr >> 5)) & 1;
            lfsr = (lfsr >> 1) | (bit << 15);
            let noise = if (lfsr & 1) != 0 { 1.0f32 } else { -1.0f32 };
            let c_drums = c_kick + noise * (snare_env * 0.18 + hat_env * 0.06);
            snare_env *= 0.9986;
            hat_env *= 0.9965;

            *s = (c_arp + c_bass + echoed + c_drums).clamp(-0.92, 0.92);
        }

        if let Ok(mut ring) = viz_pcm.lock() {
            if ring.len() + TICK_SAMPLES > 1024 {
                let drop_n = (ring.len() + TICK_SAMPLES - 1024).min(ring.len());
                ring.drain(..drop_n);
            }
            ring.extend_from_slice(&chunk);
        }
        if player.push_samples(&chunk).is_err() {
            break;
        }
        tick += 1;
        if tick >= TICKS_PER_ROW {
            tick = 0;
            row = (row + 1) % 64;
        }
    }
}
