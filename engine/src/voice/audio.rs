//! Zero-Crate macOS `AudioToolbox.framework` Real-Time `24 kHz` Streaming Speaker Output & RIFF WAV I/O.
//! Executes strictly host DAC audio playback & PCM serialization (0 neural compute on CPU).

#![allow(dead_code)]

use std::ffi::c_void;
use std::fs;
use std::path::Path;
use std::ptr;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

pub const TTS_SAMPLE_RATE: u32 = 24_000;
pub const TTS_HOP_SAMPLES: usize = 1_920; // 80 ms per 12.5 Hz codec frame
pub const ASR_SAMPLE_RATE: u32 = 16_000;

const AUDIO_FORMAT_LINEAR_PCM: u32 = 0x6c70_636d; // 'lpcm'
const LINEAR_PCM_FLAG_IS_FLOAT: u32 = 1 << 0;
const LINEAR_PCM_FLAG_IS_PACKED: u32 = 1 << 3;

const NUM_QUEUE_BUFFERS: usize = 12;
const MAX_SAMPLES_PER_BUF: usize = 16_384; // ~682 ms max per AudioQueue buffer

type AudioQueueRef = *mut c_void;

#[repr(C)]
#[derive(Clone, Copy)]
struct AudioStreamBasicDescription {
    m_sample_rate: f64,
    m_format_id: u32,
    m_format_flags: u32,
    m_bytes_per_packet: u32,
    m_frames_per_packet: u32,
    m_bytes_per_frame: u32,
    m_channels_per_frame: u32,
    m_bits_per_channel: u32,
    m_reserved: u32,
}

#[repr(C)]
struct AudioQueueBuffer {
    m_audio_data_bytes_capacity: u32,
    m_audio_data: *mut c_void,
    m_audio_data_byte_size: u32,
    m_user_data: *mut c_void,
    m_packet_description_capacity: u32,
    m_packet_descriptions: *mut c_void,
    m_packet_description_count: u32,
}

type AudioQueueOutputCallback = extern "C" fn(*mut c_void, AudioQueueRef, *mut AudioQueueBuffer);
type AudioQueueInputCallback = extern "C" fn(
    *mut c_void,
    AudioQueueRef,
    *mut AudioQueueBuffer,
    *const c_void,
    u32,
    *const c_void,
);

#[link(name = "AudioToolbox", kind = "framework")]
extern "C" {
    fn AudioQueueNewOutput(
        in_format: *const AudioStreamBasicDescription,
        in_callback_proc: AudioQueueOutputCallback,
        in_user_data: *mut c_void,
        in_callback_run_loop: *mut c_void,
        in_callback_run_loop_mode: *mut c_void,
        in_flags: u32,
        out_aq: *mut AudioQueueRef,
    ) -> i32;

    fn AudioQueueNewInput(
        in_format: *const AudioStreamBasicDescription,
        in_callback_proc: AudioQueueInputCallback,
        in_user_data: *mut c_void,
        in_callback_run_loop: *mut c_void,
        in_callback_run_loop_mode: *mut c_void,
        in_flags: u32,
        out_aq: *mut AudioQueueRef,
    ) -> i32;

    fn AudioQueueAllocateBuffer(
        in_aq: AudioQueueRef,
        in_buffer_byte_size: u32,
        out_buffer: *mut *mut AudioQueueBuffer,
    ) -> i32;

    fn AudioQueueEnqueueBuffer(
        in_aq: AudioQueueRef,
        in_buffer: *mut AudioQueueBuffer,
        in_num_packet_descs: u32,
        in_packet_descs: *const c_void,
    ) -> i32;

    fn AudioQueueStart(in_aq: AudioQueueRef, in_device_start_time: *const c_void) -> i32;
    fn AudioQueueFlush(in_aq: AudioQueueRef) -> i32;
    fn AudioQueueStop(in_aq: AudioQueueRef, in_immediate: u8) -> i32;
    fn AudioQueueDispose(in_aq: AudioQueueRef, in_immediate: u8) -> i32;
}

struct PoolState {
    free_bufs: Vec<usize>,
    in_flight: usize,
    aborted: bool,
}

struct CallbackShared {
    state: Mutex<PoolState>,
    cv: Condvar,
}

extern "C" fn aq_output_callback(
    user_data: *mut c_void,
    _aq: AudioQueueRef,
    buf: *mut AudioQueueBuffer,
) {
    if user_data.is_null() || buf.is_null() {
        return;
    }
    let shared = unsafe { &*(user_data as *const CallbackShared) };
    if let Ok(mut st) = shared.state.lock() {
        st.free_bufs.push(buf as usize);
        st.in_flight = st.in_flight.saturating_sub(1);
        shared.cv.notify_all();
    }
}

/// Zero-dependency real-time `24 kHz` mono `f32` speaker output stream via macOS `AudioToolbox`.
pub struct CoreAudioPlayer {
    aq: AudioQueueRef,
    shared: Arc<CallbackShared>,
    shared_raw: *const CallbackShared,
    started: bool,
    disposed: bool,
}

unsafe impl Send for CoreAudioPlayer {}

impl CoreAudioPlayer {
    pub fn new(sample_rate: u32) -> Result<Self, String> {
        let desc = AudioStreamBasicDescription {
            m_sample_rate: sample_rate as f64,
            m_format_id: AUDIO_FORMAT_LINEAR_PCM,
            m_format_flags: LINEAR_PCM_FLAG_IS_FLOAT | LINEAR_PCM_FLAG_IS_PACKED,
            m_bytes_per_packet: 4,
            m_frames_per_packet: 1,
            m_bytes_per_frame: 4,
            m_channels_per_frame: 1,
            m_bits_per_channel: 32,
            m_reserved: 0,
        };

        let shared = Arc::new(CallbackShared {
            state: Mutex::new(PoolState {
                free_bufs: Vec::with_capacity(NUM_QUEUE_BUFFERS),
                in_flight: 0,
                aborted: false,
            }),
            cv: Condvar::new(),
        });
        let shared_raw = Arc::into_raw(Arc::clone(&shared));

        let mut aq: AudioQueueRef = ptr::null_mut();
        let status = unsafe {
            AudioQueueNewOutput(
                &desc,
                aq_output_callback,
                shared_raw as *mut c_void,
                ptr::null_mut(),
                ptr::null_mut(),
                0,
                &mut aq,
            )
        };
        if status != 0 || aq.is_null() {
            unsafe {
                let _ = Arc::from_raw(shared_raw);
            }
            return Err(format!("AudioQueueNewOutput failed with OSStatus {}", status));
        }

        let cap_bytes = (MAX_SAMPLES_PER_BUF * 4) as u32;
        for _ in 0..NUM_QUEUE_BUFFERS {
            let mut buf_ptr: *mut AudioQueueBuffer = ptr::null_mut();
            let st = unsafe { AudioQueueAllocateBuffer(aq, cap_bytes, &mut buf_ptr) };
            if st != 0 || buf_ptr.is_null() {
                unsafe {
                    AudioQueueDispose(aq, 1);
                    let _ = Arc::from_raw(shared_raw);
                }
                return Err(format!("AudioQueueAllocateBuffer failed with OSStatus {}", st));
            }
            shared.state.lock().unwrap().free_bufs.push(buf_ptr as usize);
        }

        Ok(Self {
            aq,
            shared,
            shared_raw,
            started: false,
            disposed: false,
        })
    }

    /// Push `f32` PCM samples in `[-1.0, 1.0]` directly to the Mac's speakers.
    /// Starts hardware playback automatically on the very first chunk for minimum TTFA!
    pub fn push_samples(&mut self, samples: &[f32]) -> Result<(), String> {
        if samples.is_empty() || self.disposed {
            return Ok(());
        }
        let mut offset = 0usize;
        while offset < samples.len() {
            let take = (samples.len() - offset).min(MAX_SAMPLES_PER_BUF);
            let buf_ptr = {
                let mut st = self.shared.state.lock().unwrap();
                while st.free_bufs.is_empty() && !st.aborted {
                    let (g, _) = self.shared.cv.wait_timeout(st, Duration::from_millis(50)).unwrap();
                    st = g;
                }
                if st.aborted {
                    return Ok(());
                }
                let p = st.free_bufs.pop().unwrap() as *mut AudioQueueBuffer;
                st.in_flight += 1;
                p
            };

            unsafe {
                let dst = std::slice::from_raw_parts_mut((*buf_ptr).m_audio_data as *mut f32, take);
                for (d, &s) in dst.iter_mut().zip(&samples[offset..offset + take]) {
                    *d = s.clamp(-1.0, 1.0);
                }
                (*buf_ptr).m_audio_data_byte_size = (take * 4) as u32;

                let st = AudioQueueEnqueueBuffer(self.aq, buf_ptr, 0, ptr::null());
                if st != 0 {
                    return Err(format!("AudioQueueEnqueueBuffer failed with OSStatus {}", st));
                }
                if !self.started {
                    let start_st = AudioQueueStart(self.aq, ptr::null());
                    if start_st != 0 {
                        return Err(format!("AudioQueueStart failed with OSStatus {}", start_st));
                    }
                    self.started = true;
                }
            }
            offset += take;
        }
        Ok(())
    }

    /// Wait cleanly for all enqueued audio buffers to finish playing through the DAC.
    pub fn drain(&mut self) {
        if self.disposed || !self.started {
            return;
        }
        unsafe {
            let _ = AudioQueueFlush(self.aq);
        }
        let mut st = self.shared.state.lock().unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        while st.in_flight > 0 && !st.aborted && std::time::Instant::now() < deadline {
            let (g, _) = self.shared.cv.wait_timeout(st, Duration::from_millis(20)).unwrap();
            st = g;
        }
        drop(st);
        unsafe {
            let _ = AudioQueueStop(self.aq, 0);
        }
        self.started = false;
    }

    /// Immediately silence speaker output (`< 1 ms`, e.g. when user presses `Esc` in TUI).
    pub fn abort_immediate(&mut self) {
        if self.disposed {
            return;
        }
        if let Ok(mut st) = self.shared.state.lock() {
            st.aborted = true;
            self.shared.cv.notify_all();
        }
        unsafe {
            let _ = AudioQueueStop(self.aq, 1);
        }
        self.started = false;
    }

    fn shutdown(&mut self) {
        if self.disposed {
            return;
        }
        self.disposed = true;
        unsafe {
            let _ = AudioQueueStop(self.aq, 1);
            let _ = AudioQueueDispose(self.aq, 1);
            let _ = Arc::from_raw(self.shared_raw);
        }
    }
}

impl Drop for CoreAudioPlayer {
    fn drop(&mut self) {
        self.shutdown();
    }
}

// -----------------------------------------------------------------------------
// Zero-Crate Standard RIFF `.wav` Encoder (16-bit PCM @ 24,000 Hz Mono)
// -----------------------------------------------------------------------------

pub fn encode_wav_pcm16(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let data_bytes = (samples.len() * 2) as u32;
    let riff_size = 36u32.saturating_add(data_bytes);
    let byte_rate = sample_rate * 2;
    let mut out = Vec::with_capacity(44 + samples.len() * 2);

    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // PCM chunk size
    out.extend_from_slice(&1u16.to_le_bytes());  // AudioFormat = 1 (PCM)
    out.extend_from_slice(&1u16.to_le_bytes());  // NumChannels = 1 (Mono)
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());  // BlockAlign = 2
    out.extend_from_slice(&16u16.to_le_bytes()); // BitsPerSample = 16
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_bytes.to_le_bytes());

    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let pcm = (clamped * 32767.0).round() as i16;
        out.extend_from_slice(&pcm.to_le_bytes());
    }
    out
}

pub fn write_wav_pcm16(path: &Path, samples: &[f32], sample_rate: u32) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
    }
    let wav = encode_wav_pcm16(samples, sample_rate);
    fs::write(path, wav).map_err(|e| format!("Failed writing WAV {}: {}", path.display(), e))
}

// -----------------------------------------------------------------------------
// Zero-Crate macOS `AudioToolbox` Live `16 kHz` Microphone Recorder + VAD Telemetry
// -----------------------------------------------------------------------------

const NUM_INPUT_BUFFERS: usize = 6;
const INPUT_CHUNK_SAMPLES: usize = 800; // 50 ms @ 16,000 Hz

#[derive(Clone, Copy, Debug, Default)]
pub struct MicLiveStats {
    pub duration_sec: f64,
    pub rms: f32,
    pub peak: f32,
    pub speech_started: bool,
    pub trailing_silence_sec: f64,
    pub auto_stop_ready: bool,
}

struct InputSharedState {
    samples: Vec<f32>,
    sample_rate: u32,
    stopped: bool,
    chunks_seen: usize,
    noise_floor_rms: f32,
    last_rms: f32,
    peak: f32,
    voice_streak: usize,
    speech_started: bool,
    trailing_silence_sec: f64,
}

struct InputCallbackShared {
    state: Mutex<InputSharedState>,
}

extern "C" fn aq_input_callback(
    user_data: *mut c_void,
    in_aq: AudioQueueRef,
    in_buf: *mut AudioQueueBuffer,
    _start_time: *const c_void,
    _num_packets: u32,
    _packet_descs: *const c_void,
) {
    if user_data.is_null() || in_buf.is_null() {
        return;
    }
    let shared = unsafe { &*(user_data as *const InputCallbackShared) };
    let mut reenqueue = false;
    if let Ok(mut st) = shared.state.lock() {
        if !st.stopped {
            let byte_sz = unsafe { (*in_buf).m_audio_data_byte_size as usize };
            let n = byte_sz / 4;
            if n > 0 {
                let slice = unsafe {
                    std::slice::from_raw_parts((*in_buf).m_audio_data as *const f32, n)
                };
                st.samples.extend_from_slice(slice);

                let mut sq = 0.0f64;
                let mut pk = 0.0f32;
                for &v in slice {
                    let a = v.abs();
                    if a > pk {
                        pk = a;
                    }
                    sq += (v as f64) * (v as f64);
                }
                let rms = (sq / (n as f64)).sqrt() as f32;
                st.last_rms = rms;
                if pk > st.peak {
                    st.peak = pk;
                }
                st.chunks_seen += 1;

                // Adaptive ambient noise calibration on early quiet chunks
                if !st.speech_started && st.chunks_seen <= 5 {
                    if st.chunks_seen == 1 {
                        st.noise_floor_rms = rms.clamp(0.002, 0.025);
                    } else {
                        st.noise_floor_rms = (0.7 * st.noise_floor_rms + 0.3 * rms.min(st.noise_floor_rms * 1.5))
                            .clamp(0.002, 0.025);
                    }
                }

                let thresh = (st.noise_floor_rms * 3.2).clamp(0.011, 0.045);
                let chunk_sec = (n as f64) / (st.sample_rate.max(1) as f64);

                if rms >= thresh {
                    st.voice_streak += 1;
                    if st.voice_streak >= 2 {
                        st.speech_started = true;
                        st.trailing_silence_sec = 0.0;
                    }
                } else {
                    st.voice_streak = 0;
                    if st.speech_started && rms < thresh * 0.70 {
                        st.trailing_silence_sec += chunk_sec;
                    } else if st.speech_started {
                        st.trailing_silence_sec = (st.trailing_silence_sec - chunk_sec * 0.5).max(0.0);
                    }
                }
            }
            reenqueue = true;
        }
    }
    if reenqueue {
        unsafe {
            let _ = AudioQueueEnqueueBuffer(in_aq, in_buf, 0, ptr::null());
        }
    }
}

pub struct CoreAudioRecorder {
    aq: AudioQueueRef,
    shared: Arc<InputCallbackShared>,
    shared_raw: *const InputCallbackShared,
    disposed: bool,
}

unsafe impl Send for CoreAudioRecorder {}

impl CoreAudioRecorder {
    pub fn start(sample_rate: u32) -> Result<Self, String> {
        let desc = AudioStreamBasicDescription {
            m_sample_rate: sample_rate as f64,
            m_format_id: AUDIO_FORMAT_LINEAR_PCM,
            m_format_flags: LINEAR_PCM_FLAG_IS_FLOAT | LINEAR_PCM_FLAG_IS_PACKED,
            m_bytes_per_packet: 4,
            m_frames_per_packet: 1,
            m_bytes_per_frame: 4,
            m_channels_per_frame: 1,
            m_bits_per_channel: 32,
            m_reserved: 0,
        };

        let shared = Arc::new(InputCallbackShared {
            state: Mutex::new(InputSharedState {
                samples: Vec::with_capacity((sample_rate as usize) * 15),
                sample_rate,
                stopped: false,
                chunks_seen: 0,
                noise_floor_rms: 0.004,
                last_rms: 0.0,
                peak: 0.0,
                voice_streak: 0,
                speech_started: false,
                trailing_silence_sec: 0.0,
            }),
        });
        let shared_raw = Arc::into_raw(Arc::clone(&shared));

        let mut aq: AudioQueueRef = ptr::null_mut();
        let status = unsafe {
            AudioQueueNewInput(
                &desc,
                aq_input_callback,
                shared_raw as *mut c_void,
                ptr::null_mut(),
                ptr::null_mut(),
                0,
                &mut aq,
            )
        };
        if status != 0 || aq.is_null() {
            unsafe {
                let _ = Arc::from_raw(shared_raw);
            }
            return Err(format!(
                "AudioQueueNewInput failed with OSStatus {} (check Terminal microphone permission in macOS Privacy & Security)",
                status
            ));
        }

        let buf_bytes = (INPUT_CHUNK_SAMPLES * 4) as u32;
        for _ in 0..NUM_INPUT_BUFFERS {
            let mut buf_ptr: *mut AudioQueueBuffer = ptr::null_mut();
            let st = unsafe { AudioQueueAllocateBuffer(aq, buf_bytes, &mut buf_ptr) };
            if st != 0 || buf_ptr.is_null() {
                unsafe {
                    AudioQueueDispose(aq, 1);
                    let _ = Arc::from_raw(shared_raw);
                }
                return Err(format!("AudioQueueAllocateBuffer (input) failed with OSStatus {}", st));
            }
            let eq = unsafe { AudioQueueEnqueueBuffer(aq, buf_ptr, 0, ptr::null()) };
            if eq != 0 {
                unsafe {
                    AudioQueueDispose(aq, 1);
                    let _ = Arc::from_raw(shared_raw);
                }
                return Err(format!("AudioQueueEnqueueBuffer (input) failed with OSStatus {}", eq));
            }
        }

        let start_st = unsafe { AudioQueueStart(aq, ptr::null()) };
        if start_st != 0 {
            unsafe {
                AudioQueueDispose(aq, 1);
                let _ = Arc::from_raw(shared_raw);
            }
            return Err(format!(
                "AudioQueueStart (microphone) failed with OSStatus {} (grant Microphone access to your Terminal app)",
                start_st
            ));
        }

        Ok(Self {
            aq,
            shared,
            shared_raw,
            disposed: false,
        })
    }

    pub fn live_stats(&self) -> MicLiveStats {
        let Ok(st) = self.shared.state.lock() else {
            return MicLiveStats::default();
        };
        let duration_sec = (st.samples.len() as f64) / (st.sample_rate.max(1) as f64);
        let auto_stop_ready = st.speech_started
            && duration_sec >= 0.70
            && st.trailing_silence_sec >= 0.95
            || duration_sec >= 28.0;
        MicLiveStats {
            duration_sec,
            rms: st.last_rms,
            peak: st.peak,
            speech_started: st.speech_started,
            trailing_silence_sec: st.trailing_silence_sec,
            auto_stop_ready,
        }
    }

    /// Copy up to `max_len` most recent `16 kHz` PCM samples for real-time TUI oscilloscope/spectrum rendering.
    pub fn recent_samples(&self, max_len: usize, out: &mut Vec<f32>) {
        out.clear();
        if let Ok(st) = self.shared.state.lock() {
            let start = st.samples.len().saturating_sub(max_len);
            out.extend_from_slice(&st.samples[start..]);
        }
    }

    pub fn finish_and_take_samples(&mut self) -> Vec<f32> {
        if let Ok(mut st) = self.shared.state.lock() {
            st.stopped = true;
        }
        if !self.disposed {
            unsafe {
                let _ = AudioQueueStop(self.aq, 1);
            }
        }
        let out = self
            .shared
            .state
            .lock()
            .map(|mut st| std::mem::take(&mut st.samples))
            .unwrap_or_default();
        self.shutdown();
        out
    }

    pub fn abort(&mut self) {
        if let Ok(mut st) = self.shared.state.lock() {
            st.stopped = true;
            st.samples.clear();
        }
        self.shutdown();
    }

    fn shutdown(&mut self) {
        if self.disposed {
            return;
        }
        self.disposed = true;
        unsafe {
            let _ = AudioQueueStop(self.aq, 1);
            let _ = AudioQueueDispose(self.aq, 1);
            let _ = Arc::from_raw(self.shared_raw);
        }
    }
}

impl Drop for CoreAudioRecorder {
    fn drop(&mut self) {
        self.shutdown();
    }
}

// -----------------------------------------------------------------------------
// Zero-Crate RIFF `.wav` Reader & Windowed-Sinc Resampler -> `16,000 Hz` Mono `f32`
// -----------------------------------------------------------------------------

pub fn resample_sinc_mono(src: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == dst_rate || src.is_empty() {
        return src.to_vec();
    }
    let ratio = (dst_rate as f64) / (src_rate as f64);
    let out_len = ((src.len() as f64) * ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    let cutoff = ratio.min(1.0);
    let a = 12isize; // 24-tap Lanczos windowed sinc

    for i in 0..out_len {
        let center = (i as f64) / ratio;
        let ic = center.floor() as isize;
        let mut sum = 0.0f64;
        let mut wsum = 0.0f64;
        for k in (ic - a + 1)..=(ic + a) {
            let x = (center - (k as f64)) * cutoff;
            let w = if x.abs() < 1e-9 {
                1.0
            } else if x.abs() >= (a as f64) {
                0.0
            } else {
                let px = std::f64::consts::PI * x;
                let pa = px / (a as f64);
                (px.sin() / px) * (pa.sin() / pa)
            };
            let idx = k.clamp(0, (src.len() as isize) - 1) as usize;
            sum += (src[idx] as f64) * w;
            wsum += w;
        }
        out.push(if wsum.abs() > 1e-12 { (sum / wsum) as f32 } else { 0.0 });
    }
    out
}

pub fn parse_wav_bytes_mono_16k(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("Not a valid RIFF WAVE file".into());
    }
    let mut pos = 12usize;
    let mut audio_format = 0u16;
    let mut num_channels = 0u16;
    let mut sample_rate = 0u32;
    let mut bits_per_sample = 0u16;
    let mut data_slice: Option<&[u8]> = None;

    while pos + 8 <= bytes.len() {
        let chunk_id = &bytes[pos..pos + 4];
        let chunk_sz = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        pos += 8;
        let end = (pos + chunk_sz).min(bytes.len());
        if chunk_id == b"fmt " && end - pos >= 16 {
            audio_format = u16::from_le_bytes(bytes[pos..pos + 2].try_into().unwrap());
            num_channels = u16::from_le_bytes(bytes[pos + 2..pos + 4].try_into().unwrap());
            sample_rate = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap());
            bits_per_sample = u16::from_le_bytes(bytes[pos + 14..pos + 16].try_into().unwrap());
        } else if chunk_id == b"data" {
            data_slice = Some(&bytes[pos..end]);
            break;
        }
        pos = end + (chunk_sz & 1);
    }

    let raw = data_slice.ok_or("WAV file missing 'data' chunk")?;
    if num_channels == 0 || sample_rate == 0 {
        return Err("WAV file has invalid fmt header".into());
    }
    let ch = num_channels as usize;
    let bytes_per_smp = (bits_per_sample as usize) / 8;
    if bytes_per_smp == 0 {
        return Err(format!("Unsupported WAV bits_per_sample={}", bits_per_sample));
    }
    let frame_bytes = ch * bytes_per_smp;
    let n_frames = raw.len() / frame_bytes;
    let mut mono = Vec::with_capacity(n_frames);

    for f in 0..n_frames {
        let fbase = f * frame_bytes;
        let mut acc = 0.0f32;
        for c in 0..ch {
            let off = fbase + c * bytes_per_smp;
            let v = match (audio_format, bits_per_sample) {
                (1, 16) => {
                    let s = i16::from_le_bytes(raw[off..off + 2].try_into().unwrap());
                    (s as f32) / 32768.0
                }
                (1, 24) => {
                    let b0 = raw[off] as i32;
                    let b1 = (raw[off + 1] as i32) << 8;
                    let b2 = ((raw[off + 2] as i8) as i32) << 16;
                    ((b0 | b1 | b2) as f32) / 8388608.0
                }
                (1, 32) => {
                    let s = i32::from_le_bytes(raw[off..off + 4].try_into().unwrap());
                    (s as f32) / 2147483648.0
                }
                (3, 32) => f32::from_bits(u32::from_le_bytes(raw[off..off + 4].try_into().unwrap())),
                (fmt, bps) => {
                    return Err(format!("Unsupported WAV format={} bits={}", fmt, bps));
                }
            };
            acc += v;
        }
        mono.push(acc / (ch as f32));
    }

    Ok(resample_sinc_mono(&mono, sample_rate, ASR_SAMPLE_RATE))
}

pub fn read_wav_mono_16k(path: &Path) -> Result<Vec<f32>, String> {
    let bytes = fs::read(path).map_err(|e| format!("Failed reading WAV {}: {}", path.display(), e))?;
    parse_wav_bytes_mono_16k(&bytes)
}
