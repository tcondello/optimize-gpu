//! 1990s Demoscene Visual & Audio FX Engine (`Second Reality` / `Amiga AGA` / `Impulse Tracker` / `ACiD`).
//!
//! Zero external crates, 100% deterministic math:
//! - Half-block (`▀`) `2x` vertical subpixel TrueColor rasterizer (` plasma`, `copperbars`, `3D starfield`,
//!   `3D vector polyhedra`, `demoscene palette fire`, `Kefrens bars`, `3D horizon grid`, `DYCP sine scroller`)
//! - Real-time 16-band Hann-Goertzel spectrum analyzer with gravity-falling peak-hold caps + oscilloscope
//! - Optional 4-channel `24 kHz` procedural Amiga/SID tracker chiptune synthesizer (`/mod` / `M` in `/demo`)
//! - 4 authentic 1990s scene palettes (`SecondReality`, `AmigaCopper`, `ImpulseTracker`, `AcidAnsi`)

use std::f32::consts::TAU;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use super::terminal::{Rgb, ScreenBuf, Style};
use crate::voice::audio::{CoreAudioPlayer, TTS_SAMPLE_RATE};

const NUM_STARS: usize = 180;
const NUM_BANDS: usize = 16;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum DemoTheme {
    SecondReality,
    AmigaCopper,
    ImpulseTracker,
    AcidAnsi,
}

impl DemoTheme {
    pub(super) fn next(self) -> Self {
        match self {
            Self::SecondReality => Self::AmigaCopper,
            Self::AmigaCopper => Self::ImpulseTracker,
            Self::ImpulseTracker => Self::AcidAnsi,
            Self::AcidAnsi => Self::SecondReality,
        }
    }

    pub(super) fn from_arg(arg: &str) -> Option<Self> {
        match arg.trim().to_ascii_lowercase().as_str() {
            "1" | "sr" | "second_reality" | "secondreality" | "fc" | "futurecrew" => {
                Some(Self::SecondReality)
            }
            "2" | "amiga" | "copper" | "amiga_copper" | "aga" => Some(Self::AmigaCopper),
            "3" | "it" | "ft2" | "tracker" | "impulse" | "impulse_tracker" => {
                Some(Self::ImpulseTracker)
            }
            "4" | "acid" | "ansi" | "bbs" | "vga" | "acid_ansi" => Some(Self::AcidAnsi),
            _ => None,
        }
    }

    pub(super) fn label(self) -> &'static str {
        match self {
            Self::SecondReality => "SECOND REALITY '93",
            Self::AmigaCopper => "AMIGA AGA COPPER '92",
            Self::ImpulseTracker => "IMPULSE TRACKER '95",
            Self::AcidAnsi => "ACiD VGA ANSI '94",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum FxLevel {
    Full,
    Minimal,
    Off,
}

impl FxLevel {
    pub(super) fn cycle(self) -> Self {
        match self {
            Self::Full => Self::Minimal,
            Self::Minimal => Self::Off,
            Self::Off => Self::Full,
        }
    }

    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Full => "FULL 60FPS",
            Self::Minimal => "MINIMAL",
            Self::Off => "STATIC",
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct Palette {
    pub bg_void: Rgb,
    pub bg_header: Rgb,
    pub bg_deck: Rgb,
    pub bg_footer: Rgb,
    pub bevel_hi: Rgb,
    pub bevel_mid: Rgb,
    pub bevel_dim: Rgb,
    pub copper_a: Rgb,
    pub copper_b: Rgb,
    pub copper_c: Rgb,
    pub text_hi: Rgb,
    pub text_muted: Rgb,
    pub text_dim: Rgb,
    pub badge_user: Rgb,
    pub badge_bot: Rgb,
}

pub(super) fn theme_palette(theme: DemoTheme) -> Palette {
    match theme {
        DemoTheme::SecondReality => Palette {
            bg_void: Rgb(7, 9, 18),
            bg_header: Rgb(13, 18, 34),
            bg_deck: Rgb(10, 15, 28),
            bg_footer: Rgb(12, 16, 30),
            bevel_hi: Rgb(76, 224, 255),
            bevel_mid: Rgb(46, 98, 164),
            bevel_dim: Rgb(26, 48, 86),
            copper_a: Rgb(0, 232, 255),
            copper_b: Rgb(245, 52, 188),
            copper_c: Rgb(255, 206, 56),
            text_hi: Rgb(234, 242, 255),
            text_muted: Rgb(128, 154, 192),
            text_dim: Rgb(74, 94, 128),
            badge_user: Rgb(20, 74, 126),
            badge_bot: Rgb(18, 92, 86),
        },
        DemoTheme::AmigaCopper => Palette {
            bg_void: Rgb(10, 7, 24),
            bg_header: Rgb(22, 15, 44),
            bg_deck: Rgb(16, 11, 34),
            bg_footer: Rgb(19, 13, 38),
            bevel_hi: Rgb(255, 174, 68),
            bevel_mid: Rgb(148, 78, 162),
            bevel_dim: Rgb(68, 38, 92),
            copper_a: Rgb(255, 116, 28),
            copper_b: Rgb(255, 234, 142),
            copper_c: Rgb(162, 68, 255),
            text_hi: Rgb(248, 238, 228),
            text_muted: Rgb(176, 144, 192),
            text_dim: Rgb(104, 82, 126),
            badge_user: Rgb(118, 48, 32),
            badge_bot: Rgb(76, 38, 118),
        },
        DemoTheme::ImpulseTracker => Palette {
            bg_void: Rgb(10, 14, 19),
            bg_header: Rgb(20, 28, 38),
            bg_deck: Rgb(14, 20, 28),
            bg_footer: Rgb(18, 25, 34),
            bevel_hi: Rgb(255, 214, 92),
            bevel_mid: Rgb(68, 122, 138),
            bevel_dim: Rgb(36, 62, 76),
            copper_a: Rgb(68, 238, 128),
            copper_b: Rgb(255, 210, 74),
            copper_c: Rgb(68, 198, 234),
            text_hi: Rgb(232, 240, 232),
            text_muted: Rgb(134, 164, 160),
            text_dim: Rgb(78, 102, 102),
            badge_user: Rgb(32, 78, 96),
            badge_bot: Rgb(28, 86, 56),
        },
        DemoTheme::AcidAnsi => Palette {
            bg_void: Rgb(4, 4, 22),
            bg_header: Rgb(0, 22, 58),
            bg_deck: Rgb(6, 10, 36),
            bg_footer: Rgb(0, 18, 46),
            bevel_hi: Rgb(85, 255, 255),
            bevel_mid: Rgb(170, 42, 185),
            bevel_dim: Rgb(24, 76, 148),
            copper_a: Rgb(85, 255, 255),
            copper_b: Rgb(255, 85, 255),
            copper_c: Rgb(255, 255, 85),
            text_hi: Rgb(255, 255, 255),
            text_muted: Rgb(170, 196, 228),
            text_dim: Rgb(85, 112, 158),
            badge_user: Rgb(0, 85, 170),
            badge_bot: Rgb(128, 0, 128),
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

#[inline]
fn add_rgb(a: Rgb, b: Rgb) -> Rgb {
    Rgb(a.0.saturating_add(b.0), a.1.saturating_add(b.1), a.2.saturating_add(b.2))
}

pub(super) fn copper_gradient(pal: &Palette, phase: f32) -> Rgb {
    let u = phase.rem_euclid(1.0);
    if u < 0.333 {
        lerp_rgb(pal.copper_a, pal.copper_b, u * 3.0)
    } else if u < 0.666 {
        lerp_rgb(pal.copper_b, pal.copper_c, (u - 0.333) * 3.0)
    } else {
        lerp_rgb(pal.copper_c, pal.copper_a, (u - 0.666) * 3.0)
    }
}

#[derive(Clone, Copy)]
struct Star3D {
    x: f32,
    y: f32,
    z: f32,
}

pub(super) struct BannerInfo<'a> {
    pub model_name: &'a str,
    pub is_inkling: bool,
    pub gguf_size_gb: f64,
    pub active_sms: u32,
    pub ping_ms: f64,
    pub effort: &'a str,
    pub spec_on: bool,
    pub gamma: usize,
    pub last_tps: f64,
}

pub(super) struct DemosceneEngine {
    t0: Instant,
    last_step: Instant,
    time_s: f32,
    theme: DemoTheme,
    fx: FxLevel,
    showcase_open: bool,
    showcase_scene: usize,
    showcase_auto: bool,
    showcase_scene_t0: Instant,

    stars: [Star3D; NUM_STARS],
    rng_state: u64,

    bands: [f32; NUM_BANDS],
    peaks: [f32; NUM_BANDS],
    peak_vel: [f32; NUM_BANDS],
    scope: [f32; 64],
    bus_energy: f32,

    fire_w: usize,
    fire_h: usize,
    fire_buf: Vec<u8>,

    pix_w: usize,
    pix_h: usize,
    pixels: Vec<Rgb>,

    scroller_pos: f32,
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
            theme: DemoTheme::SecondReality,
            fx: FxLevel::Full,
            showcase_open: false,
            showcase_scene: 0,
            showcase_auto: true,
            showcase_scene_t0: now,
            stars: [Star3D { x: 0.0, y: 0.0, z: 1.0 }; NUM_STARS],
            rng_state: 0x9E37_79B9_7F4A_7C15,
            bands: [0.0; NUM_BANDS],
            peaks: [0.0; NUM_BANDS],
            peak_vel: [0.0; NUM_BANDS],
            scope: [0.0; 64],
            bus_energy: 0.0,
            fire_w: 0,
            fire_h: 0,
            fire_buf: Vec::new(),
            pix_w: 0,
            pix_h: 0,
            pixels: Vec::new(),
            scroller_pos: 0.0,
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

    pub(super) fn status_tags(&self) -> (&'static str, &'static str, bool, bool) {
        (
            self.theme.label(),
            self.fx.label(),
            self.showcase_open,
            self.mod_playing.load(Ordering::Relaxed),
        )
    }

    pub(super) fn configure(&mut self, cmd: &str, arg: &str, viz_pcm: &Arc<Mutex<Vec<f32>>>) -> String {
        match cmd {
            "theme_cycle" => {
                self.theme = self.theme.next();
                format!("♦ Palette switched to [{}]", self.theme.label())
            }
            "theme" => {
                if arg.is_empty() {
                    self.theme = self.theme.next();
                } else if let Some(t) = DemoTheme::from_arg(arg) {
                    self.theme = t;
                } else {
                    return "Themes: second_reality (sr), amiga_copper (aga), impulse_tracker (it), acid_ansi (ansi)".into();
                }
                format!("♦ Demoscene palette -> [{}]", self.theme.label())
            }
            "fx" => {
                self.fx = match arg.trim().to_ascii_lowercase().as_str() {
                    "full" | "on" | "60" => FxLevel::Full,
                    "min" | "minimal" | "low" => FxLevel::Minimal,
                    "off" | "static" | "0" => FxLevel::Off,
                    _ => self.fx.cycle(),
                };
                format!("♦ Demoscene FX mode -> [{}]", self.fx.label())
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
                    "♦ Entered 60 FPS MEGADEMO (1-4/Space: Scene • T: Palette • M: Chiptune • Esc: Return)".into()
                } else {
                    "♦ Returned to RTX Chat TUI".into()
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
                    "♫ Playing 'GA102 OVERDRIVE.MOD' (4-ch 24kHz Amiga/SID Tracker Synth • /mod off or Esc stops)".into()
                } else if !want_on && is_on {
                    self.stop_mod_synth();
                    "♫ Stopped tracker chiptune synth".into()
                } else {
                    "♫ Tracker chiptune already active".into()
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

    pub(super) fn pulse_token(&mut self, tok_id: i32, live_tps: f64) {
        let boost = ((live_tps / 140.0) as f32).clamp(0.25, 1.0);
        self.bus_energy = (self.bus_energy * 0.55 + boost * 0.65).clamp(0.0, 1.0);
        let h = (tok_id as u32).wrapping_mul(2654435761);
        let b0 = (h as usize) % NUM_BANDS;
        let b1 = ((h >> 8) as usize) % NUM_BANDS;
        let b2 = ((h >> 16) as usize) % NUM_BANDS;
        for &(bi, k) in &[(b0, 0.92f32), (b1, 0.74f32), (b2, 0.58f32)] {
            let v = (boost * k).clamp(0.15, 1.0);
            if v > self.bands[bi] {
                self.bands[bi] = v;
            }
            if v > self.peaks[bi] {
                self.peaks[bi] = v;
                self.peak_vel[bi] = 0.0;
            }
        }
    }

    pub(super) fn step(
        &mut self,
        pcm_window: Option<(&[f32], u32)>,
        is_streaming: bool,
        chat_empty: bool,
    ) -> bool {
        let now = Instant::now();
        let dt = now.duration_since(self.last_step).as_secs_f32().clamp(0.001, 0.10);
        self.last_step = now;

        let mod_on = self.mod_playing.load(Ordering::Relaxed);
        let has_audio = pcm_window.map_or(false, |(s, _)| s.len() >= 64);
        let active_anim = self.showcase_open
            || mod_on
            || has_audio
            || is_streaming
            || self.fx == FxLevel::Full;

        if !active_anim && self.fx == FxLevel::Off {
            return false;
        }

        self.time_s = self.t0.elapsed().as_secs_f32();
        let warp = if is_streaming { 2.4 } else if mod_on || has_audio { 1.6 } else { 1.0 };

        // Advance 3D warp starfield when showcase or attract banner is visible
        if self.showcase_open || (chat_empty && self.fx != FxLevel::Off) {
            for i in 0..NUM_STARS {
                self.stars[i].z -= dt * 0.72 * warp;
                if self.stars[i].z <= 0.06 {
                    self.stars[i] = self.spawn_star(false);
                }
            }
        }

        // Auto-advance megademo scene every 11s unless manually pinned
        if self.showcase_open && self.showcase_auto && self.showcase_scene_t0.elapsed() > Duration::from_secs(11) {
            self.showcase_scene = (self.showcase_scene + 1) % 4;
            self.showcase_scene_t0 = now;
        }

        // Advance smooth marquee scroller
        if self.fx == FxLevel::Full || self.showcase_open || is_streaming || mod_on {
            self.scroller_pos += dt * 16.0;
        }

        // Update 16-band Goertzel spectrum + oscilloscope
        if let Some((samples, sr)) = pcm_window.filter(|(s, _)| s.len() >= 64) {
            self.update_spectrum_from_pcm(samples, sr);
        } else {
            self.bus_energy *= 0.88;
            for i in 0..NUM_BANDS {
                let idle_wave = if self.fx == FxLevel::Full || self.showcase_open {
                    let s = (self.time_s * 2.3 + (i as f32) * 0.48).sin() * 0.5 + 0.5;
                    0.04 + 0.09 * s
                } else {
                    0.0
                };
                self.bands[i] = (self.bands[i] * 0.80).max(idle_wave + self.bus_energy * 0.25);
                self.peak_vel[i] += dt * 0.55;
                self.peaks[i] = (self.peaks[i] - self.peak_vel[i] * dt * 2.2).max(self.bands[i]);
                if self.peaks[i] <= self.bands[i] + 0.001 {
                    self.peak_vel[i] = 0.0;
                }
            }
            for (k, v) in self.scope.iter_mut().enumerate() {
                let ph = self.time_s * 5.0 + (k as f32) * 0.22;
                *v = (ph.sin() * 0.5 + (ph * 2.1).cos() * 0.3) * self.bus_energy;
            }
        }

        active_anim
    }

    fn update_spectrum_from_pcm(&mut self, samples: &[f32], sample_rate: u32) {
        let win_len = samples.len().min(256);
        let tail = &samples[samples.len() - win_len..];
        let sr = (sample_rate.max(8_000)) as f32;

        // Capture 64-point decimated oscilloscope trace
        for i in 0..64 {
            let idx = (i * win_len / 64).min(win_len - 1);
            self.scope[i] = (tail[idx] * 2.4).clamp(-1.0, 1.0);
        }

        // 16 log-spaced Goertzel resonators (75 Hz .. 6,600 Hz)
        for b in 0..NUM_BANDS {
            let frac = (b as f32) / ((NUM_BANDS - 1) as f32);
            let freq = 75.0 * (6600.0f32 / 75.0).powf(frac);
            if freq >= sr * 0.47 {
                self.bands[b] *= 0.82;
                continue;
            }
            let omega = TAU * freq / sr;
            let coeff = 2.0 * omega.cos();
            let mut s1 = 0.0f32;
            let mut s2 = 0.0f32;
            let n_inv = 1.0 / (win_len as f32);
            for (n, &smp) in tail.iter().enumerate() {
                let hann = 0.5 - 0.5 * (TAU * (n as f32) * n_inv).cos();
                let s0 = smp * hann + coeff * s1 - s2;
                s2 = s1;
                s1 = s0;
            }
            let power = (s1 * s1 + s2 * s2 - coeff * s1 * s2).max(0.0);
            let mag = (power.sqrt() * n_inv * (6.5 + frac * 11.0)).sqrt();
            let level = mag.clamp(0.0, 1.0);
            if level >= self.bands[b] {
                self.bands[b] = self.bands[b] * 0.32 + level * 0.68;
            } else {
                self.bands[b] = self.bands[b] * 0.78 + level * 0.22;
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
    // Chrome Helpers: Copper Bevel Bars, Inline Spectrum, Marquee, Attract & Demo
    // =========================================================================

    pub(super) fn draw_chrome_bars(
        &self,
        scr: &mut ScreenBuf,
        w: usize,
        divider_y: usize,
        footer_y: usize,
        marquee_msg: &str,
    ) {
        let pal = self.palette();

        // Row `divider_y`: Animated Amiga Copperbar Bevel Divider (`╠═══...═══╣`)
        if divider_y < scr.dims().1 && w >= 8 {
            scr.put_char(0, divider_y, '╠', Style::new(pal.bevel_hi, pal.bg_void).bold());
            for x in 1..w.saturating_sub(1) {
                let u = (x as f32) / (w.max(1) as f32);
                let wave = (u * 3.5 - self.time_s * 0.85).sin() * 0.5 + 0.5;
                let col = if self.fx == FxLevel::Off {
                    pal.bevel_mid
                } else {
                    let c = copper_gradient(&pal, u * 1.4 - self.time_s * 0.28);
                    lerp_rgb(pal.bevel_dim, c, 0.35 + 0.55 * wave)
                };
                let ch = if wave > 0.86 && self.fx != FxLevel::Off { '═' } else { '═' };
                scr.put_char(x, divider_y, ch, Style::new(col, pal.bg_void));
            }
            scr.put_char(
                w.saturating_sub(1),
                divider_y,
                '╣',
                Style::new(pal.bevel_hi, pal.bg_void).bold(),
            );
        }

        // Row `footer_y`: Smooth 1990s Demoscene Copper Scroller (`░▒▓ ... ▓▒░`)
        if !marquee_msg.is_empty() && footer_y < scr.dims().1 && w >= 16 {
            scr.fill_row(footer_y, pal.bg_footer);
            scr.put_str(0, footer_y, w, "░▒▓", Style::new(pal.copper_a, pal.bg_footer));
            scr.put_str(
                w.saturating_sub(3),
                footer_y,
                w,
                "▓▒░",
                Style::new(pal.copper_b, pal.bg_footer),
            );
            let inner_w = w.saturating_sub(8);
            let chars: Vec<char> = marquee_msg.chars().collect();
            if !chars.is_empty() && inner_w > 4 {
                let base = if self.fx == FxLevel::Off {
                    0usize
                } else {
                    (self.scroller_pos as usize) % chars.len()
                };
                for i in 0..inner_w {
                    let ch = chars[(base + i) % chars.len()];
                    let edge_fade = (i.min(inner_w - 1 - i) as f32 / 6.0).clamp(0.25, 1.0);
                    let c_base = copper_gradient(&pal, (i as f32) / 48.0 - self.time_s * 0.22);
                    let fg = scale_rgb(lerp_rgb(pal.text_hi, c_base, 0.45), edge_fade);
                    scr.put_char(4 + i, footer_y, ch, Style::new(fg, pal.bg_footer));
                }
            }
        }
    }

    pub(super) fn draw_spectrum_strip(
        &self,
        scr: &mut ScreenBuf,
        x: usize,
        y: usize,
        max_x: usize,
        compact: bool,
    ) -> usize {
        const BARS: [char; 9] = ['·', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
        let pal = self.palette();
        let bg = if compact { pal.bg_header } else { pal.bg_deck };
        let n = if compact { 12usize } else { NUM_BANDS };
        if x + n + 2 >= max_x {
            return x;
        }
        let mut cx = scr.put_char(x, y, '[', Style::new(pal.bevel_mid, bg));
        for i in 0..n {
            let bi = i * NUM_BANDS / n;
            let lvl = self.bands[bi].clamp(0.0, 1.0);
            let pk = self.peaks[bi].clamp(0.0, 1.0);
            let idx = ((lvl * 8.0).round() as usize).min(8);
            let ch = if idx == 0 && pk > 0.18 { '˙' } else { BARS[idx] };
            let fg = if lvl > 0.80 {
                Rgb(255, 78, 92)
            } else if lvl > 0.52 {
                Rgb(255, 214, 68)
            } else if lvl > 0.14 {
                Rgb(72, 242, 136)
            } else {
                pal.bevel_mid
            };
            cx += scr.put_char(cx, y, ch, Style::new(fg, bg).bold());
        }
        cx += scr.put_char(cx, y, ']', Style::new(pal.bevel_mid, bg));

        // Optional mini oscilloscope trace when wide enough in non-compact deck
        if !compact && cx + 20 < max_x {
            cx += scr.put_str(cx, y, max_x, " ~", Style::new(pal.bevel_dim, bg));
            for k in 0..16 {
                let s = self.scope[k * 4];
                let ch = if s > 0.45 {
                    '▀'
                } else if s > 0.12 {
                    '─'
                } else if s > -0.20 {
                    '·'
                } else {
                    '▄'
                };
                cx += scr.put_char(cx, y, ch, Style::new(pal.copper_a, bg));
            }
        }
        cx
    }

    pub(super) fn draw_attract_or_showcase(
        &mut self,
        scr: &mut ScreenBuf,
        x0: usize,
        y0: usize,
        w: usize,
        h: usize,
        info: &BannerInfo<'_>,
    ) {
        if self.showcase_open {
            let (sw, sh) = scr.dims();
            self.render_megademo_fullscreen(scr, sw, sh, info);
        } else {
            self.render_attract_home(scr, x0, y0, w, h, info);
        }
    }

    // =========================================================================
    // Subpixel (`▀`) Canvas Helpers
    // =========================================================================

    fn ensure_canvas(&mut self, pw: usize, ph: usize, clear: Rgb) {
        self.pix_w = pw;
        self.pix_h = ph;
        self.pixels.clear();
        self.pixels.resize(pw * ph, clear);
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
        let mut guard = 0usize;
        loop {
            self.put_px(x0, y0, col);
            if x0 == x1 && y0 == y1 || guard > 1024 {
                break;
            }
            guard += 1;
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

    fn blit_subpixels(&self, scr: &mut ScreenBuf, x0: usize, y0: usize, cols: usize, rows: usize) {
        for r in 0..rows {
            let py_top = r * 2;
            let py_bot = py_top + 1;
            for c in 0..cols {
                let top = if py_top < self.pix_h && c < self.pix_w {
                    self.pixels[py_top * self.pix_w + c]
                } else {
                    Rgb(0, 0, 0)
                };
                let bot = if py_bot < self.pix_h && c < self.pix_w {
                    self.pixels[py_bot * self.pix_w + c]
                } else {
                    top
                };
                if top == bot {
                    scr.put_char(x0 + c, y0 + r, ' ', Style::new(top, top));
                } else {
                    scr.put_char(x0 + c, y0 + r, '▀', Style::new(top, bot));
                }
            }
        }
    }

    fn render_starfield_px(&mut self, pal: &Palette) {
        let pw = self.pix_w as f32;
        let ph = self.pix_h as f32;
        let cx = pw * 0.5;
        let cy = ph * 0.5;
        for i in 0..NUM_STARS {
            let st = self.stars[i];
            let inv_z = 1.0 / st.z.max(0.06);
            let sx = (cx + st.x * inv_z * (pw * 0.36)) as isize;
            let sy = (cy + st.y * inv_z * (ph * 0.42)) as isize;
            let bright = ((2.1 - st.z) / 2.0).clamp(0.15, 1.0);
            let col = lerp_rgb(pal.bevel_dim, Rgb(245, 250, 255), bright);
            self.put_px(sx, sy, col);
            if st.z < 0.55 {
                self.put_px(sx + 1, sy, scale_rgb(col, 0.65));
            }
        }
    }

    fn render_3d_polyhedron_px(&mut self, cx: f32, cy: f32, scale: f32, pal: &Palette) {
        // Rotating 3D Stellated Cube / Octahedron ("Future Crew Glenz Vector")
        const VERTS: [(f32, f32, f32); 14] = [
            // Inner cube
            (-1.0, -1.0, -1.0),
            (1.0, -1.0, -1.0),
            (1.0, 1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, 1.0),
            (-1.0, 1.0, 1.0),
            // 6 octahedron apexes
            (0.0, 0.0, -1.75),
            (0.0, 0.0, 1.75),
            (-1.75, 0.0, 0.0),
            (1.75, 0.0, 0.0),
            (0.0, -1.75, 0.0),
            (0.0, 1.75, 0.0),
        ];
        const EDGES: [(usize, usize); 24] = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
            (8, 0), (8, 1), (8, 2), (8, 3),
            (9, 4), (9, 5), (9, 6), (9, 7),
            (10, 11), (12, 13), (8, 13), (9, 12),
        ];

        let t = self.time_s;
        let (ax, ay, az) = (t * 0.92, t * 1.27, t * 0.61);
        let (sx, cx_r) = ax.sin_cos();
        let (sy, cy_r) = ay.sin_cos();
        let (sz, cz_r) = az.sin_cos();

        let mut proj = [(0isize, 0isize, 0.0f32); 14];
        for (i, &(vx, vy, vz)) in VERTS.iter().enumerate() {
            // Rotate X
            let y1 = vy * cx_r - vz * sx;
            let z1 = vy * sx + vz * cx_r;
            // Rotate Y
            let x2 = vx * cy_r + z1 * sy;
            let z2 = -vx * sy + z1 * cy_r;
            // Rotate Z
            let x3 = x2 * cz_r - y1 * sz;
            let y3 = x2 * sz + y1 * cz_r;

            let dist = 4.2 + z2;
            let p = scale / dist.max(1.2);
            // Terminal cells are ~2x taller than wide, and subpixels are ~1:1 square!
            let px = (cx + x3 * p * 1.15).round() as isize;
            let py = (cy + y3 * p).round() as isize;
            proj[i] = (px, py, z2);
        }

        // Draw back-to-front depth-cued wireframe edges
        for (ei, &(a, b)) in EDGES.iter().enumerate() {
            let (x0, y0, za) = proj[a];
            let (x1, y1, zb) = proj[b];
            let depth = ((2.0 - (za + zb) * 0.5) / 3.6).clamp(0.20, 1.0);
            let base_c = if ei >= 12 { pal.copper_b } else { pal.copper_a };
            let col = lerp_rgb(pal.bevel_dim, base_c, depth);
            self.draw_line_px(x0, y0, x1, y1, col);
        }
        // Bright vertex glints
        for &(px, py, z) in &proj[..8] {
            if z < 0.8 {
                self.put_px(px, py, Rgb(255, 255, 255));
            }
        }
    }

    fn draw_big_text_px(&mut self, text: &str, x0: isize, y0: isize, scale: isize, pal: &Palette, dycp_amp: f32) {
        let mut pen_x = x0;
        for (ci, ch) in text.chars().enumerate() {
            let bits = glyph_5x5(ch);
            for gx in 0..5isize {
                let col_x = pen_x + gx * scale;
                let wave_y = if dycp_amp > 0.0 {
                    ((col_x as f32) * 0.085 + self.time_s * 4.4).sin() * dycp_amp
                } else {
                    0.0
                } as isize;
                for gy in 0..5isize {
                    let bit = (bits >> ((4 - gy) * 5 + (4 - gx))) & 1;
                    if bit != 0 {
                        let c_phase = (ci as f32) * 0.11 + (gy as f32) * 0.14 - self.time_s * 0.55;
                        let base = copper_gradient(pal, c_phase);
                        // Chrome specular highlight across middle scanline (`gy == 2`)
                        let col = if gy == 1 || gy == 2 {
                            lerp_rgb(base, Rgb(255, 255, 255), 0.52)
                        } else {
                            base
                        };
                        for sy in 0..scale {
                            for sx in 0..scale {
                                self.put_px(col_x + sx, y0 + wave_y + gy * scale + sy, col);
                            }
                        }
                        // Crisp 1px drop shadow for 90s bitmap logo pop
                        self.put_px(col_x + scale, y0 + wave_y + gy * scale + scale, Rgb(4, 6, 14));
                    }
                }
            }
            pen_x += 6 * scale;
        }
    }

    // =========================================================================
    // Empty-Chat Attract Screen: Subpixel Intro Banner + Authentic `.NFO` Card
    // =========================================================================

    fn render_attract_home(
        &mut self,
        scr: &mut ScreenBuf,
        x0: usize,
        y0: usize,
        w: usize,
        h: usize,
        info: &BannerInfo<'_>,
    ) {
        let pal = self.palette();
        if w < 36 || h < 6 {
            return;
        }

        let box_x = x0 + 1;
        let box_w = w.saturating_sub(2);
        let canvas_rows = if h >= 22 {
            9usize
        } else if h >= 16 {
            6usize
        } else {
            4usize
        }
        .min(h.saturating_sub(2));

        // Frame top border of the Demoscene Intro Window
        scr.put_char(box_x, y0, '╔', Style::new(pal.bevel_hi, pal.bg_void).bold());
        let hdr_lbl = format!(
            "═[ ♦ OPTIMIZE-GPU CREW PRESENTS :: {} // {} ]",
            if info.is_inkling { "INKLING-2 LOOPED MoE" } else { "QWEN3 MEGAKERNEL" },
            self.theme.label()
        );
        let mut hx = scr.put_str(box_x + 1, y0, box_x + box_w - 1, &hdr_lbl, Style::new(pal.copper_c, pal.bg_void).bold());
        while hx < box_x + box_w - 1 {
            hx += scr.put_char(hx, y0, '═', Style::new(pal.bevel_mid, pal.bg_void));
        }
        scr.put_char(box_x + box_w - 1, y0, '╗', Style::new(pal.bevel_hi, pal.bg_void).bold());

        // Subpixel canvas inside `[box_x + 1 .. box_x + box_w - 1, y0 + 1 .. y0 + 1 + canvas_rows]`
        let cw = box_w.saturating_sub(2);
        let pw = cw;
        let ph = canvas_rows * 2;
        self.ensure_canvas(pw, ph, pal.bg_void);

        // 1. Subtle background Amiga raster plasma ribbons + 3D starfield
        for py in 0..ph {
            let yf = (py as f32) / (ph.max(1) as f32);
            let bar1 = ((yf - (0.50 + 0.30 * (self.time_s * 1.7).sin())).abs() * 9.5).min(1.0);
            let bar2 = ((yf - (0.50 + 0.28 * (self.time_s * 1.3 + 2.1).sin())).abs() * 10.5).min(1.0);
            for px in 0..pw {
                let xf = (px as f32) / (pw.max(1) as f32);
                let pl = ((xf * 6.0 + self.time_s * 1.4).sin()
                    + (yf * 5.0 - self.time_s * 1.1).cos())
                    * 0.25
                    + 0.5;
                let mut col = lerp_rgb(pal.bg_void, scale_rgb(pal.bevel_dim, 0.55), pl * 0.55);
                if bar1 < 1.0 {
                    let k = (1.0 - bar1) * (1.0 - bar1) * 0.48;
                    col = add_rgb(col, scale_rgb(pal.copper_a, k));
                }
                if bar2 < 1.0 {
                    let k = (1.0 - bar2) * (1.0 - bar2) * 0.42;
                    col = add_rgb(col, scale_rgb(pal.copper_b, k));
                }
                self.pixels[py * pw + px] = col;
            }
        }
        self.render_starfield_px(&pal);

        // 2. Left/Center Chrome Pixel Masthead (`RTX 3080`) + Right 3D Vector Glenz Polyhedron
        let poly_cx = if pw >= 74 { (pw as f32) * 0.80 } else { (pw as f32) * 0.50 };
        let poly_cy = (ph as f32) * 0.50;
        let poly_scale = (ph as f32) * 1.25;
        self.render_3d_polyhedron_px(poly_cx, poly_cy, poly_scale, &pal);

        if pw >= 56 && ph >= 10 {
            let logo_scale = if pw >= 96 && ph >= 16 { 2 } else { 1 };
            let logo_txt = "RTX 3080";
            let logo_y = ((ph as isize) - 5 * logo_scale) / 2;
            self.draw_big_text_px(logo_txt, 4, logo_y.max(1), logo_scale, &pal, 1.4);
        }

        for r in 0..canvas_rows {
            scr.put_char(box_x, y0 + 1 + r, '║', Style::new(pal.bevel_mid, pal.bg_void));
            scr.put_char(
                box_x + box_w - 1,
                y0 + 1 + r,
                '║',
                Style::new(pal.bevel_mid, pal.bg_void),
            );
        }
        self.blit_subpixels(scr, box_x + 1, y0 + 1, cw, canvas_rows);

        let mid_y = y0 + 1 + canvas_rows;
        if mid_y >= y0 + h {
            return;
        }

        // `.NFO` Release Info Card directly integrated into the lower half of the frame!
        let rem_rows = (y0 + h).saturating_sub(mid_y + 1);
        if rem_rows < 3 {
            scr.put_char(box_x, mid_y, '╚', Style::new(pal.bevel_hi, pal.bg_void).bold());
            for x in (box_x + 1)..(box_x + box_w - 1) {
                scr.put_char(x, mid_y, '═', Style::new(pal.bevel_mid, pal.bg_void));
            }
            scr.put_char(box_x + box_w - 1, mid_y, '╝', Style::new(pal.bevel_hi, pal.bg_void).bold());
            return;
        }

        scr.put_char(box_x, mid_y, '╠', Style::new(pal.bevel_hi, pal.bg_void).bold());
        let nfo_hdr = "═[ FILE_ID.DIZ & RELEASE.NFO // 100% GA102 SM_86 • ZERO DOCKER • ZERO CRATES ]";
        let mut nx = scr.put_str(
            box_x + 1,
            mid_y,
            box_x + box_w - 1,
            nfo_hdr,
            Style::new(pal.copper_a, pal.bg_void).bold(),
        );
        while nx < box_x + box_w - 1 {
            nx += scr.put_char(nx, mid_y, '═', Style::new(pal.bevel_mid, pal.bg_void));
        }
        scr.put_char(box_x + box_w - 1, mid_y, '╣', Style::new(pal.bevel_hi, pal.bg_void).bold());

        let mode_str = if info.is_inkling {
            if info.spec_on {
                format!("DSpark Block-8 Speculative (gamma={}, ~143 tok/s)", info.gamma)
            } else {
                format!("48-Layer Persistent B=1 Megakernel (~122 tok/s • Effort:{})", info.effort)
            }
        } else if info.spec_on {
            "BM=4 Lossless Speculative Megakernel (~270 tok/s)".to_string()
        } else {
            "T=1 68-SM Persistent Megakernel (~173 tok/s)".to_string()
        };
        let voice_str = if info.is_inkling {
            "Co-Resident Qwen3-ASR-0.6B + Inkling-2-Mini + Qwen3-TTS-0.6B (24kHz Code2Wav)"
        } else {
            "Co-Resident Qwen3-ASR-1.7B + Qwen3 + Qwen3-TTS-1.7B (24kHz Code2Wav)"
        };

        let tps_note = if info.last_tps > 0.0 {
            format!(" • {:.1} tok/s", info.last_tps)
        } else {
            String::new()
        };
        let nfo_lines: [(Rgb, String); 8] = [
            (
                pal.text_hi,
                format!(
                    "  ► MODEL    : {} ({:.2} GB • {} SMs active • {:.1}ms IPC{})",
                    info.model_name, info.gguf_size_gb, info.active_sms, info.ping_ms, tps_note
                ),
            ),
            (pal.copper_c, format!("  ► KERNEL   : {}", mode_str)),
            (pal.copper_a, format!("  ► AUDIO    : {}", voice_str)),
            (pal.bevel_dim, "  ──────────────────────────────────────────────────────────────────────────────".into()),
            (
                pal.text_hi,
                "  [Ctrl+R] /listen : Speak into Mac Mic -> RTX 3080 STT -> LLM -> 24kHz TTS".into(),
            ),
            (
                pal.text_muted,
                "  [/talk]          : Hands-free voice loop   │ [/effort 0|0.2|0.9] : Thinking depth".into(),
            ),
            (
                pal.copper_b,
                "  [Ctrl+D] /demo   : 60FPS Fullscreen Demo!  │ [/mod]              : 4-Ch Tracker Synth!".into(),
            ),
            (
                pal.text_muted,
                "  [Ctrl+T] /theme  : Cycle 1990s Palettes    │ [/bench] [/spec] [/fx] [/help]".into(),
            ),
        ];

        let max_body = rem_rows.saturating_sub(1).min(nfo_lines.len());
        for i in 0..max_body {
            let ry = mid_y + 1 + i;
            scr.fill_row(ry, pal.bg_void);
            scr.put_char(box_x, ry, '║', Style::new(pal.bevel_mid, pal.bg_void));
            let (fg, ref txt) = nfo_lines[i];
            scr.put_str(box_x + 1, ry, box_x + box_w - 1, txt, Style::new(fg, pal.bg_void));
            scr.put_char(box_x + box_w - 1, ry, '║', Style::new(pal.bevel_mid, pal.bg_void));
        }

        let bot_y = mid_y + 1 + max_body;
        if bot_y < y0 + h {
            scr.put_char(box_x, bot_y, '╚', Style::new(pal.bevel_hi, pal.bg_void).bold());
            for x in (box_x + 1)..(box_x + box_w - 1) {
                scr.put_char(x, bot_y, '═', Style::new(pal.bevel_mid, pal.bg_void));
            }
            scr.put_char(box_x + box_w - 1, bot_y, '╝', Style::new(pal.bevel_hi, pal.bg_void).bold());
        }
    }

    // =========================================================================
    // Fullscreen 60 FPS 4-Part 1990s Megademo (`/demo` / `Ctrl+D`)
    // =========================================================================

    fn render_megademo_fullscreen(&mut self, scr: &mut ScreenBuf, w: usize, h: usize, info: &BannerInfo<'_>) {
        let pal = self.palette();
        scr.resize_and_clear(w, h, pal.bg_void);
        if w < 24 || h < 10 {
            return;
        }

        let scene_names = [
            "PART 1/4 :: TRIGONOMETRIC PLASMA & AMIGA COPPERBARS",
            "PART 2/4 :: 3D GLENZ VECTOR & PARALLAX STARFIELD",
            "PART 3/4 :: DOOM PALETTE FIRE & KEFRENS RASTERS",
            "PART 4/4 :: DYCP SINE SCROLLER & 3D HORIZON GRID",
        ];

        // Top HUD bar (Row 0)
        scr.fill_row(0, pal.bg_header);
        let hud_l = format!(
            " ♦ RTX MEGADEMO '93 // {} ",
            scene_names[self.showcase_scene % 4]
        );
        let mut x = scr.put_str(0, 0, w, &hud_l, Style::new(pal.bg_void, pal.copper_a).bold());
        let mod_tag = if self.mod_playing.load(Ordering::Relaxed) {
            " [♫ MOD:ON] "
        } else {
            " [M:PLAY MOD] "
        };
        x = scr.put_str(x + 1, 0, w, mod_tag, Style::new(pal.copper_c, pal.bg_header).bold());
        let hud_r = "[1-4/SPC]:SCENE  [T]:PALETTE  [M]:CHIPTUNE  [ESC]:CHAT ";
        let rx = w.saturating_sub(hud_r.chars().count());
        if rx > x + 1 {
            scr.put_str(rx, 0, w, hud_r, Style::new(pal.text_hi, pal.bg_header));
        }

        let canvas_rows = h.saturating_sub(2);
        let pw = w;
        let ph = canvas_rows * 2;
        self.ensure_canvas(pw, ph, pal.bg_void);

        match self.showcase_scene % 4 {
            0 => self.scene_plasma_and_copper(&pal),
            1 => self.scene_vector_glenz_and_stars(&pal),
            2 => self.scene_doom_fire_and_kefrens(&pal),
            _ => self.scene_dycp_and_horizon(&pal, info),
        }

        self.blit_subpixels(scr, 0, 1, w, canvas_rows);

        // Bottom status bar (`h - 1`)
        let fy = h - 1;
        scr.fill_row(fy, pal.bg_footer);
        let sc_txt = format!(
            "░▒▓ GA102 ({} SMs) • {} • PALETTE: {} • 0 DOCKER • 0 CRATES • PRESS [M] FOR 4-CH TRACKER SYNTH • [ESC] RETURN TO CHAT   ",
            info.active_sms,
            info.model_name,
            self.theme.label()
        );
        self.draw_chrome_bars(scr, w, h + 1, fy, &sc_txt);
    }

    fn scene_plasma_and_copper(&mut self, pal: &Palette) {
        let pw = self.pix_w;
        let ph = self.pix_h;
        let t = self.time_s;

        // Precompute 5 intertwining Amiga copperbar vertical centers
        let mut bars = [(0.0f32, Rgb(0, 0, 0)); 5];
        for i in 0..5 {
            let fi = i as f32;
            let yc = 0.50 + 0.36 * (t * (1.25 + fi * 0.21) + fi * 1.15).sin();
            let col = copper_gradient(pal, fi * 0.22 - t * 0.15);
            bars[i] = (yc * (ph as f32), col);
        }

        for py in 0..ph {
            let yf = (py as f32) * 0.065;
            for px in 0..pw {
                let xf = (px as f32) * 0.045;
                let v1 = (xf + t * 1.6).sin();
                let v2 = (yf - t * 1.3).sin();
                let v3 = ((xf + yf) * 0.7 + t * 1.1).sin();
                let cx = xf - (pw as f32) * 0.0225;
                let cy = yf - (ph as f32) * 0.0325;
                let v4 = ((cx * cx + cy * cy).sqrt() * 1.4 - t * 2.1).sin();
                let u = (v1 + v2 + v3 + v4) * 0.125 + 0.5;
                let mut col = scale_rgb(copper_gradient(pal, u), 0.55);

                // Overlay metallic Amiga copperbars with white-hot specular core
                for &(by, bc) in &bars {
                    let dist = ((py as f32) - by).abs();
                    if dist < 4.2 {
                        let k = 1.0 - dist / 4.2;
                        let spec = if dist < 1.1 {
                            lerp_rgb(bc, Rgb(255, 255, 255), 0.78)
                        } else {
                            scale_rgb(bc, k * k)
                        };
                        col = lerp_rgb(col, spec, k);
                    }
                }
                self.pixels[py * pw + px] = col;
            }
        }
        if pw >= 68 && ph >= 20 {
            self.draw_big_text_px("SECOND REALITY", 6, 4, 1, pal, 2.0);
        }
    }

    fn scene_vector_glenz_and_stars(&mut self, pal: &Palette) {
        self.render_starfield_px(pal);
        let pw = self.pix_w as f32;
        let ph = self.pix_h as f32;
        let cx = pw * 0.5;
        let cy = ph * 0.5;

        // Outer Lissajous orbital energy rings
        let steps = 96usize;
        let mut prev: Option<(isize, isize)> = None;
        for i in 0..=steps {
            let a = (i as f32 / steps as f32) * TAU;
            let rx = (a * 2.0 + self.time_s * 1.1).cos() * (pw * 0.36);
            let ry = (a * 3.0 - self.time_s * 1.4).sin() * (ph * 0.34);
            let px = (cx + rx) as isize;
            let py = (cy + ry) as isize;
            if let Some((ox, oy)) = prev {
                let c = scale_rgb(copper_gradient(pal, i as f32 / steps as f32), 0.55);
                self.draw_line_px(ox, oy, px, py, c);
            }
            prev = Some((px, py));
        }

        let scale = ph.min(pw * 0.65) * 1.35;
        self.render_3d_polyhedron_px(cx, cy, scale, pal);
    }

    fn scene_doom_fire_and_kefrens(&mut self, pal: &Palette) {
        let pw = self.pix_w;
        let ph = self.pix_h;
        if self.fire_w != pw || self.fire_h != ph || self.fire_buf.len() != pw * ph {
            self.fire_w = pw;
            self.fire_h = ph;
            self.fire_buf.clear();
            self.fire_buf.resize(pw * ph, 0);
        }

        // Seed bottom generator row with roaring heat + audio bass boost!
        let bass = (self.bands[0] + self.bands[1] + self.bands[2]) * 0.33;
        if ph >= 2 {
            for x in 0..pw {
                let r = self.next_rand_f32();
                let heat = if r < 0.82 + bass * 0.15 { 255u8 } else { 140u8 };
                self.fire_buf[(ph - 1) * pw + x] = heat;
            }
        }

        // Classic 1993 cellular upward convection + wind cooling
        for y in 0..ph.saturating_sub(1) {
            for x in 0..pw {
                let drift = ((self.next_rand_f32() * 3.0) as usize).saturating_sub(1);
                let sx = (x + drift).min(pw - 1);
                let below = self.fire_buf[(y + 1) * pw + sx] as i32;
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

        // Overlay classic Amiga Kefrens vertical raster column wave in upper sky
        for y in 0..(ph * 2 / 3) {
            let yf = y as f32 * 0.11;
            let cx = ((pw as f32) * 0.5
                + ((pw as f32) * 0.30) * (self.time_s * 2.3 + yf * 0.55).sin()
                + ((pw as f32) * 0.12) * (self.time_s * 3.7 - yf * 0.9).cos()) as isize;
            for dx in -5isize..=5isize {
                let k = 1.0 - (dx.abs() as f32) / 5.5;
                let rc = lerp_rgb(pal.copper_a, Rgb(255, 255, 255), k * k);
                self.put_px(cx + dx, y as isize, rc);
            }
        }
    }

    fn scene_dycp_and_horizon(&mut self, pal: &Palette, info: &BannerInfo<'_>) {
        let pw = self.pix_w;
        let ph = self.pix_h;
        let horizon = ph * 52 / 100;

        // Upper sky starfield
        self.render_starfield_px(pal);

        // Lower 3D perspective synth/Amiga checkerboard floor
        for py in horizon..ph {
            let dy = ((py - horizon + 1) as f32) / ((ph - horizon).max(1) as f32);
            let z = 1.0 / dy.max(0.04);
            let fog = dy.powf(0.75);
            for px in 0..pw {
                let nx = ((px as f32) / (pw.max(1) as f32) - 0.5) * 2.0;
                let wx = nx * z * 2.6;
                let wz = z * 1.4 + self.time_s * 4.5;
                let checker = ((wx.floor() as i32) ^ (wz.floor() as i32)) & 1;
                let tile = if checker == 0 { pal.copper_a } else { pal.bg_header };
                self.pixels[py * pw + px] = lerp_rgb(pal.bg_void, tile, fog * 0.72);
            }
        }
        // Bright horizon laser line
        if horizon < ph {
            for px in 0..pw {
                self.pixels[horizon * pw + px] = pal.copper_b;
            }
        }

        // Giant scrolling DYCP sine-wave bitmap banner across the sky!
        let msg = format!(
            "   RTX 3080 GA102 • 68 SMs • {} • ZERO DOCKER • ZERO CRATES • GREETZ FUTURE CREW & TRITON!   ",
            info.model_name.to_ascii_uppercase()
        );
        let scale = if ph >= 36 { 2isize } else { 1isize };
        let char_px = 6 * scale;
        let total_px = (msg.chars().count() as isize) * char_px;
        let scroll_x = -(((self.time_s * 42.0 * (scale as f32)) as isize).rem_euclid(total_px.max(1)));
        let base_y = (horizon as isize / 2 - 3 * scale).max(2);
        self.draw_big_text_px(&msg, scroll_x, base_y, scale, pal, 4.0 * (scale as f32));
        self.draw_big_text_px(&msg, scroll_x + total_px, base_y, scale, pal, 4.0 * (scale as f32));

        // 16-band subpixel spectrum bars rising from the bottom of the screen!
        let bar_area_w = (pw * 3 / 4).max(16);
        let x_start = (pw - bar_area_w) / 2;
        let bw = (bar_area_w / NUM_BANDS).max(1);
        let max_bh = (ph / 3).max(6);
        for b in 0..NUM_BANDS {
            let bh = ((self.bands[b] * (max_bh as f32)).round() as usize).min(max_bh);
            let pk_y = ph.saturating_sub(1 + ((self.peaks[b] * (max_bh as f32)).round() as usize).min(max_bh));
            for bx in 0..bw.saturating_sub(1).max(1) {
                let x = x_start + b * bw + bx;
                for by in 0..bh {
                    let y = ph - 1 - by;
                    let ratio = (by as f32) / (max_bh as f32);
                    let c = if ratio > 0.75 {
                        Rgb(255, 72, 92)
                    } else if ratio > 0.45 {
                        Rgb(255, 214, 64)
                    } else {
                        Rgb(64, 244, 132)
                    };
                    self.put_px(x as isize, y as isize, c);
                }
                self.put_px(x as isize, pk_y as isize, Rgb(255, 255, 255));
            }
        }
    }
}

// =============================================================================
// Compact 5x5 Demoscene Pixel Font (Bit-Packed `u32`, MSB = top-left `(0,0)`)
// =============================================================================

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
        'J' => 0b00111_00010_00010_10010_01100,
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
        '.' => 0b00000_00000_00000_00000_00100,
        ':' => 0b00000_00100_00000_00100_00000,
        '!' => 0b00100_00100_00100_00000_00100,
        '&' => 0b01100_10010_01101_10010_01101,
        '•' | '*' => 0b00000_00100_01110_00100_00000,
        _ => 0,
    }
}

// =============================================================================
// Procedural 4-Channel 24 kHz Amiga/SID Tracker Chiptune ("GA102 OVERDRIVE.MOD")
// =============================================================================

#[inline]
fn midi_to_hz(note: i32) -> f32 {
    440.0 * 2.0f32.powf((note - 69) as f32 / 12.0)
}

fn run_tracker_chiptune_loop(
    player: &mut CoreAudioPlayer,
    cancel: &AtomicBool,
    viz_pcm: &Arc<Mutex<Vec<f32>>>,
) {
    // 125 BPM, 16th-note row = 120 ms = 2,880 samples @ 24 kHz (sub-stepped in 480-sample 50Hz Amiga ticks!)
    const TICK_SAMPLES: usize = 480; // 20 ms = 50 Hz PAL Amiga VBlank tick!
    const TICKS_PER_ROW: usize = 6;  // Classic ProTracker `Speed 6` @ 125 BPM!
    const CHORDS: [[i32; 3]; 8] = [
        [60, 63, 67], // Cm
        [60, 63, 67], // Cm
        [56, 60, 63], // Ab
        [63, 67, 70], // Eb
        [58, 62, 65], // Bb
        [53, 56, 60], // Fm
        [55, 59, 62], // G
        [55, 62, 65], // G7
    ];
    const BASS_ROOTS: [i32; 8] = [36, 36, 32, 39, 34, 29, 31, 31];
    // 64-row lead melody (0 = rest/sustain)
    const LEAD: [i32; 64] = [
        72, 0, 75, 74, 72, 0, 67, 0,  72, 74, 75, 79, 77, 75, 74, 72,
        68, 0, 72, 0,  75, 0, 79, 77, 75, 0,  74, 0,  70, 0,  67, 0,
        70, 0, 74, 75, 77, 0, 74, 0,  65, 0,  68, 72, 75, 74, 72, 68,
        67, 0, 71, 74, 79, 0, 77, 74, 75, 74, 72, 71, 72, 0,  67, 0,
    ];

    let sr_inv = 1.0 / (TTS_SAMPLE_RATE as f32);
    let mut ph_arp = 0.0f32;
    let mut ph_bass = 0.0f32;
    let mut ph_lead = 0.0f32;
    let mut ph_kick = 0.0f32;
    let mut lfsr = 0xACE1u32;
    let mut delay_ring = vec![0.0f32; 5760]; // 240ms dotted-8th tracker echo
    let mut delay_idx = 0usize;

    let mut row = 0usize;
    let mut tick = 0usize;
    let mut lead_note = 72i32;
    let mut lead_env = 0.0f32;
    let mut bass_env = 0.0f32;
    let mut kick_env = 0.0f32;
    let mut snare_env = 0.0f32;
    let mut hat_env = 0.0f32;
    let mut chunk = [0.0f32; TICK_SAMPLES];

    while !cancel.load(Ordering::Relaxed) {
        let bar = (row / 8) % 8;
        let step_in_bar = row % 8;
        let chord = CHORDS[bar];
        let arp_semi = chord[tick % 3];

        if tick == 0 {
            // Trigger row events
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

        let bass_note = BASS_ROOTS[bar] + if step_in_bar % 2 == 1 { 12 } else { 0 };
        let f_arp = midi_to_hz(arp_semi);
        let f_bass = midi_to_hz(bass_note);
        let f_lead = midi_to_hz(lead_note);
        let pwm = 0.30 + 0.22 * ((row as f32) * 0.18).sin();

        for s in chunk.iter_mut() {
            // Ch 1: 50Hz Amiga `0xy` PWM Chord Arpeggio
            ph_arp = (ph_arp + f_arp * sr_inv).fract();
            let c_arp = if ph_arp < pwm { 0.14 } else { -0.14 };

            // Ch 2: Punchy Octave Saw/Pulse Bass
            ph_bass = (ph_bass + f_bass * sr_inv).fract();
            let raw_bass = (ph_bass * 2.0 - 1.0) * 0.6 + if ph_bass < 0.5 { 0.4 } else { -0.4 };
            let c_bass = raw_bass * bass_env * 0.24;
            bass_env *= 0.99935;

            // Ch 3: Vibrato Square Lead + Dotted-8th Echo
            let vib = 1.0 + 0.0045 * (ph_arp * TAU).sin();
            ph_lead = (ph_lead + f_lead * vib * sr_inv).fract();
            let dry_lead = (if ph_lead < 0.42 { 1.0 } else { -1.0 }) * lead_env * 0.16;
            lead_env *= 0.99965;
            let echoed = dry_lead + delay_ring[delay_idx] * 0.38;
            delay_ring[delay_idx] = echoed;
            delay_idx = (delay_idx + 1) % delay_ring.len();

            // Ch 4: Pitch-Swept Kick + LFSR Noise Snare/Hi-Hat
            let f_kick = 44.0 + 135.0 * kick_env * kick_env;
            ph_kick = (ph_kick + f_kick * sr_inv).fract();
            let c_kick = (ph_kick * TAU).sin() * kick_env * 0.30;
            kick_env *= 0.9982;

            let bit = ((lfsr >> 0) ^ (lfsr >> 2) ^ (lfsr >> 3) ^ (lfsr >> 5)) & 1;
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
